from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from jobs.eval import eval_checkpoint_generation as semantic_eval
from jobs.eval import score_chk3_native_checkpoint_sweep_semantic as subject
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


class FakeBERT:
    def __init__(self, *, nonfinite: bool = False) -> None:
        self.nonfinite = nonfinite
        self.calls: list[tuple[list[str], list[str]]] = []

    def score(self, candidates, references):
        self.calls.append((list(candidates), list(references)))
        size = len(candidates)
        f1 = [float("nan")] * size if self.nonfinite else [0.8] * size
        return {
            "bertscore_precision": [0.7] * size,
            "bertscore_recall": [0.9] * size,
            "bertscore_f1": f1,
        }

    def semantic_metadata(self):
        return {"chunk_audit": {"silent_truncation": False}}


class FakeMPNet:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    def score(self, candidates, references):
        self.calls.append((list(candidates), list(references)))
        return [0.6] * len(candidates)

    def semantic_metadata(self):
        return {"chunk_audit": {"silent_truncation": False}}


def _row(index: int, *, valid: bool = True) -> dict:
    source_analysis = f"Source analysis reports level {index}."
    answer = f"candidate answer reports level {index}."
    reference = f"reference Minutes {index}"
    return {
        "sample_id": f"sample-{index:02d}",
        "length_bucket": ("short", "medium", "long")[index // 4],
        "source_prompt_sha256": f"{index + 1:064x}",
        "source_analysis_sha256": f"{index + 21:064x}",
        "reference_minutes_sha256": f"{index + 41:064x}",
        "completion_sha256": f"{index + 61:064x}",
        "answer_sha256": f"{index + 81:064x}",
        "analysis_reference_exact_identity": index == 0,
        "normalized_identity": index == 0,
        "generated_text": f"reasoning\n</think>\n{answer}",
        "source_analysis": source_analysis,
        "answer": answer,
        "reference_minutes": reference,
        "hit_eos": valid,
        "cap_reached": not valid,
        "think_boundary_count": 1,
        "exact_boundary_delimiter_count": 1,
        "final_answer_single_paragraph": True,
        "full_token_4gram_repetition": 0.0,
        "tail_token_4gram_repetition": 0.0,
        "strict_periodic_tail": False,
        "quality_valid": valid,
        "quality_failures": [] if valid else ["missing_terminal_eos"],
    }


def _run(run_id: str, *, invalid_index: int | None = None) -> dict:
    baseline = run_id == subject.BASELINE_RUN_ID
    step = None if baseline else int(run_id.removeprefix("cp"))
    rows = [
        _row(index, valid=index != invalid_index)
        for index in range(subject.EXPECTED_SAMPLE_COUNT)
    ]
    return {
        "run_id": run_id,
        "role": "baseline" if baseline else "candidate",
        "checkpoint_step": step,
        "stage_id": "chk1" if baseline else "chk3",
        "model_label": run_id,
        "manifest_binding": {
            "path": f"/{run_id}/manifest.json",
            "sha256": "a" * 64,
            "payload_sha256": "b" * 64,
        },
        "manifest": {
            "model": {"path": f"/{run_id}/model", "files": {"x": "c" * 64}},
            "adapter": (
                None
                if baseline
                else {
                    "path": f"/adapters/checkpoint-{step}",
                    "files": {"x": "d" * 64},
                    "declared_base_model_path": f"/{run_id}/model",
                }
            ),
            "artifacts": {"generations": {"path": f"/{run_id}/generations.jsonl"}},
        },
        "results": rows,
        "generation_contract": {"generation_mode": "greedy", "max_new_tokens": 3072},
    }


def _merged_run(run_id: str, *, invalid_index: int | None = None) -> dict:
    step = int(run_id.removeprefix("cp"))
    run = _run(run_id, invalid_index=invalid_index)
    run["model_label"] = f"chk3-{run_id}-exact-merged"
    run["manifest"]["adapter"] = None
    run["manifest"]["model"] = {
        "path": f"/models/chk3_{run_id}_merged",
        "files": {"model.safetensors": f"{step:064x}"},
    }
    return run


def _runs(*, cp200_invalid: int | None = 1) -> dict[str, dict]:
    result = {subject.BASELINE_RUN_ID: _run(subject.BASELINE_RUN_ID)}
    for step in subject.REQUIRED_CHECKPOINT_STEPS:
        run_id = f"cp{step}"
        result[run_id] = _run(
            run_id, invalid_index=cp200_invalid if step == 200 else None
        )
    return result


def _build(*, bert=None, runs=None, **kwargs):
    return subject.build_semantic_sweep(
        runs=_runs() if runs is None else runs,
        bert=FakeBERT() if bert is None else bert,
        mpnet=FakeMPNet(),
        semantic_provenance={"binding": {"sha256": "e" * 64}},
        sample_manifest_binding={"sha256": "f" * 64},
        source_bindings={"semantic_scorer": {"sha256": "1" * 64}},
        **kwargs,
    )


def test_sealed_sweep_has_zero_penalty_and_three_explicit_cohorts() -> None:
    result = _build()
    assert validate_manifest_integrity(result) == result["integrity"][
        "payload_sha256"
    ]
    assert len(result["row_scores"]) == 5 * 12
    failed = next(
        row
        for row in result["row_scores"]
        if row["run_id"] == "cp200" and row["sample_id"] == "sample-01"
    )
    assert failed["raw_metrics"] is None
    assert failed["penalized_metrics"] == {
        metric: 0.0 for metric in subject.METRICS
    }
    assert failed["penalty_reasons"] == ["delivery_invalid"]

    cohorts = result["summaries"]["cp200"]["cohorts"]
    assert set(cohorts) == set(subject.COHORTS)
    assert cohorts["all_n12"]["cases"] == 12
    assert cohorts["all_n12"]["penalized_cases"] == 1
    assert cohorts["all_n12"]["metrics"]["bertscore_f1"]["mean"] == pytest.approx(
        11 * 0.8 / 12
    )
    assert cohorts["non_identity_n11"]["cases"] == 11
    assert cohorts["non_identity_n11"]["penalized_cases"] == 1
    assert cohorts["valid_only"]["cases"] == 11
    assert cohorts["valid_only"]["metrics"]["bertscore_f1"]["mean"] == 0.8
    assert result["paired_contrasts_vs_chk1"]["cp200"]["all_n12"][
        "candidate_minus_chk1"
    ]["bertscore_f1"]["mean"] == pytest.approx(-0.8 / 12)


def test_only_delivery_and_recomputed_core_hard_valid_answers_reach_encoders() -> None:
    bert = FakeBERT()
    result = _build(bert=bert)
    assert len(bert.calls) == 1
    candidates, references = bert.calls[0]
    assert len(candidates) == 59
    assert len(references) == 59
    assert "candidate answer reports level 1." in candidates
    assert result["semantic_backend_audit"]["scored_pairs"] == 59


def test_cp200_59bb_terminal_period_signed_surface_false_positive_is_repaired() -> None:
    runs = _runs(cp200_invalid=None)
    sample_id = "chk1-analysis-2023-07-26-59bbaffd1cb36046"
    for run in runs.values():
        run["results"][1]["sample_id"] = sample_id

    row = runs["cp200"]["results"][1]
    source = "West Texas Intermediate rose from $70.66 to $74.17."
    answer = (
        "West Texas Intermediate rose from $70.66 to $74.17, while other "
        "conditions were unchanged."
    )
    row.update(
        {
            "source_analysis": source,
            "answer": answer,
            "generated_text": f"reasoning\n</think>\n{answer}",
            "quality_valid": False,
            "quality_failures": ["signed_numeric_surface_not_preserved"],
        }
    )

    result = _build(runs=runs)
    scored = next(
        item
        for item in result["row_scores"]
        if item["run_id"] == "cp200" and item["sample_id"] == sample_id
    )
    assert scored["stored_runner_quality_valid"] is False
    assert scored["recomputed_core_hard_valid"] is True
    assert scored["semantic_eligible"] is True
    assert scored["penalty_reasons"] == []
    assert scored["raw_metrics"] is not None
    signed = scored["recomputed_core_metrics"]["corrected_signed_number_metrics"]
    assert signed["preserved"] is True
    assert signed["missing_occurrences"] == 0
    assert signed["unsupported_occurrences"] == 0
    discrepancy = scored["stored_vs_recomputed"]
    assert discrepancy == {
        "validity_changed": True,
        "failure_set_changed": True,
        "removed_stored_failures": ["signed_numeric_surface_not_preserved"],
        "added_recomputed_failures": [],
    }
    cp200_all = result["summaries"]["cp200"]["cohorts"]["all_n12"]
    assert cp200_all["penalized_cases"] == 0
    assert cp200_all["stored_runner_quality_valid_cases"] == 11
    assert cp200_all["recomputed_core_hard_valid_cases"] == 12
    assert cp200_all["stored_vs_recomputed_discrepancy_cases"] == 1


def test_corrected_signed_surface_still_rejects_a_real_numeric_change() -> None:
    metrics = subject._corrected_signed_number_metrics(
        "The value ended at $74.17.",
        "The value ended at $74.18,",
    )
    assert metrics["preserved"] is False
    assert metrics["missing_surfaces"] == [["74.17", 1]]
    assert metrics["unsupported_surfaces"] == [["74.18", 1]]


def test_nonfinite_semantic_result_fails_closed() -> None:
    with pytest.raises(subject.NativeSweepSemanticError, match="finite similarity"):
        _build(bert=FakeBERT(nonfinite=True))


def test_candidate_inventory_and_unexpected_runs_fail_closed() -> None:
    runs = _runs()
    del runs["cp318"]
    with pytest.raises(subject.NativeSweepSemanticError, match="inventory"):
        _build(runs=runs)


def test_exact_merged_finalists_build_sealed_n12_n11_and_chk1_contrasts() -> None:
    runs = {
        subject.BASELINE_RUN_ID: _run(subject.BASELINE_RUN_ID),
        "cp200": _merged_run("cp200", invalid_index=2),
        "cp318": _merged_run("cp318"),
    }
    result = _build(
        runs=runs,
        candidate_steps=subject.MERGED_FINALIST_CHECKPOINT_STEPS,
        evaluation_id=subject.MERGED_FINALIST_EVALUATION_ID,
        candidate_representation="exact_merged",
    )
    assert validate_manifest_integrity(result) == result["integrity"][
        "payload_sha256"
    ]
    assert result["evaluation_id"] == subject.MERGED_FINALIST_EVALUATION_ID
    assert result["candidate_checkpoints"] == [200, 318]
    assert result["candidate_representation"] == "exact_merged"
    assert len(result["row_scores"]) == 36
    assert set(result["paired_contrasts_vs_chk1"]) == {"cp200", "cp318"}
    assert result["summaries"]["cp200"]["cohorts"]["all_n12"][
        "penalized_cases"
    ] == 1
    assert result["summaries"]["cp200"]["cohorts"]["non_identity_n11"][
        "cases"
    ] == 11


def test_merged_signed_surface_occurrence_is_diagnostic_not_preregistered_gate() -> None:
    runs = {
        subject.BASELINE_RUN_ID: _run(subject.BASELINE_RUN_ID),
        "cp200": _merged_run("cp200"),
        "cp318": _merged_run("cp318"),
    }
    row = runs["cp318"]["results"][1]
    source = (
        "Growth slowed in the first quarter of 2024 before rebounding in the "
        "second quarter."
    )
    answer = (
        "Growth slowed in the first quarter of 2024 before rebounding in the "
        "second quarter of 2024."
    )
    row.update(
        {
            "source_analysis": source,
            "answer": answer,
            "generated_text": f"reasoning\n</think>\n{answer}",
            "quality_valid": True,
            "quality_failures": [],
        }
    )
    result = _build(
        runs=runs,
        candidate_steps=subject.MERGED_FINALIST_CHECKPOINT_STEPS,
        evaluation_id=subject.MERGED_FINALIST_EVALUATION_ID,
        candidate_representation="exact_merged",
    )
    scored = next(
        item
        for item in result["row_scores"]
        if item["run_id"] == "cp318" and item["sample_id"] == "sample-01"
    )
    assert scored["preregistered_core_valid"] is True
    assert scored["extended_signed_surface_valid"] is False
    assert scored["strict_extended_core_valid"] is False
    assert scored["semantic_eligibility_basis"] == "preregistered_core"
    assert scored["semantic_eligible"] is True
    assert scored["penalty_reasons"] == []
    assert scored["raw_metrics"] is not None
    cp318_all = result["summaries"]["cp318"]["cohorts"]["all_n12"]
    assert cp318_all["penalized_cases"] == 0
    assert cp318_all["preregistered_core_valid_cases"] == 12
    assert cp318_all["extended_signed_surface_valid_cases"] == 11
    runs = _runs()
    runs["cp999"] = copy.deepcopy(runs["cp318"])
    runs["cp999"]["run_id"] = "cp999"
    with pytest.raises(subject.NativeSweepSemanticError, match="unexpected run"):
        _build(runs=runs)


def test_atomic_output_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "semantic.json"
    subject._write_new_sealed_json(output, {"first": True})
    with pytest.raises(subject.NativeSweepSemanticError, match="refusing to overwrite"):
        subject._write_new_sealed_json(output, {"second": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"first": True}


def test_checkpoint_step_comes_from_sealed_adapter_path() -> None:
    run = _run("cp200")
    assert subject._checkpoint_step(run) == 200
    run["manifest"]["adapter"]["path"] = "/adapters/not-a-checkpoint"
    with pytest.raises(subject.NativeSweepSemanticError, match="not a checkpoint"):
        subject._checkpoint_step(run)


def test_merged_checkpoint_step_requires_no_adapter_and_unique_supported_label() -> None:
    run = _merged_run("cp200")
    assert subject._merged_checkpoint_step(run) == 200
    run["manifest"]["adapter"] = {"path": "/adapters/checkpoint-200"}
    with pytest.raises(subject.NativeSweepSemanticError, match="adapter overlay"):
        subject._merged_checkpoint_step(run)
    run = _merged_run("cp200")
    run["model_label"] = "chk3-cp200-cp318-exact-merged"
    with pytest.raises(subject.NativeSweepSemanticError, match="exactly one"):
        subject._merged_checkpoint_step(run)


def test_loader_deep_validates_every_run_and_rejects_contract_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_load(path, *, expected_stage_id, **_kwargs):
        stem = Path(path).stem
        calls.append((stem, expected_stage_id))
        if expected_stage_id == "chk1":
            run = _run(subject.BASELINE_RUN_ID)
        else:
            run = _run(stem)
        run["manifest"]["model"] = {
            "path": "/shared/chk1",
            "files": {"x": "c" * 64},
        }
        if run["manifest"]["adapter"] is not None:
            run["manifest"]["adapter"]["declared_base_model_path"] = "/shared/chk1"
        return run

    monkeypatch.setattr(subject.native_eval, "load_and_validate_run", fake_load)
    candidate_paths = [tmp_path / f"cp{step}.json" for step in subject.REQUIRED_CHECKPOINT_STEPS]
    runs = subject.load_sweep_runs(
        candidate_manifests=candidate_paths,
        baseline_manifest=tmp_path / "baseline.json",
        sample_manifest=tmp_path / "samples.json",
        sample_manifest_sha256="a" * 64,
    )
    assert list(runs) == [subject.BASELINE_RUN_ID, "cp170", "cp200", "cp230", "cp318"]
    assert calls == [
        ("cp170", "chk3"),
        ("cp200", "chk3"),
        ("cp230", "chk3"),
        ("cp318", "chk3"),
        ("baseline", "chk1"),
    ]

    def fake_drift(path, *, expected_stage_id, **kwargs):
        run = fake_load(path, expected_stage_id=expected_stage_id, **kwargs)
        if Path(path).stem == "cp230":
            run["generation_contract"] = {"generation_mode": "greedy", "max_new_tokens": 1}
        return run

    monkeypatch.setattr(subject.native_eval, "load_and_validate_run", fake_drift)
    with pytest.raises(subject.NativeSweepSemanticError, match="contract drift"):
        subject.load_sweep_runs(
            candidate_manifests=candidate_paths,
            baseline_manifest=None,
            sample_manifest=tmp_path / "samples.json",
            sample_manifest_sha256="a" * 64,
        )


def test_merged_finalist_loader_deep_validates_cp200_cp318_and_chk1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_load(path, *, expected_stage_id, **_kwargs):
        stem = Path(path).stem
        calls.append((stem, expected_stage_id))
        if expected_stage_id == "chk1":
            return _run(subject.BASELINE_RUN_ID)
        return _merged_run(stem)

    monkeypatch.setattr(subject.native_eval, "load_and_validate_run", fake_load)
    runs = subject.load_merged_finalist_runs(
        finalist_manifests=[tmp_path / "cp318.json", tmp_path / "cp200.json"],
        baseline_manifest=tmp_path / "baseline.json",
        sample_manifest=tmp_path / "samples.json",
        sample_manifest_sha256="a" * 64,
    )
    assert list(runs) == [subject.BASELINE_RUN_ID, "cp200", "cp318"]
    assert calls == [
        ("cp318", "chk3"),
        ("cp200", "chk3"),
        ("baseline", "chk1"),
    ]
    assert runs["cp200"]["manifest"]["adapter"] is None
    assert runs["cp318"]["checkpoint_step"] == 318

    with pytest.raises(subject.NativeSweepSemanticError, match="two distinct"):
        subject.load_merged_finalist_runs(
            finalist_manifests=[tmp_path / "cp200.json"],
            baseline_manifest=tmp_path / "baseline.json",
            sample_manifest=tmp_path / "samples.json",
            sample_manifest_sha256="a" * 64,
        )


def test_cli_exposes_mutually_exclusive_merged_finalist_mode(tmp_path: Path) -> None:
    common = [
        "--baseline-manifest",
        str(tmp_path / "baseline.json"),
        "--sample-manifest",
        str(tmp_path / "samples.json"),
        "--sample-manifest-sha256",
        "a" * 64,
        "--semantic-manifest",
        str(tmp_path / "semantic.json"),
        "--output",
        str(tmp_path / "result.json"),
    ]
    args = subject.build_parser().parse_args(
        [
            "--merged-finalist-manifest",
            str(tmp_path / "cp200.json"),
            "--merged-finalist-manifest",
            str(tmp_path / "cp318.json"),
            *common,
        ]
    )
    assert [path.name for path in args.merged_finalist_manifest] == [
        "cp200.json",
        "cp318.json",
    ]
    assert args.candidate_manifest is None


def test_merged_finalist_score_mode_requires_chk1_baseline(tmp_path: Path) -> None:
    with pytest.raises(subject.NativeSweepSemanticError, match="requires.*baseline"):
        subject.score_sweep(
            candidate_manifests=(),
            merged_finalist_manifests=(
                tmp_path / "cp200.json",
                tmp_path / "cp318.json",
            ),
            baseline_manifest=None,
            sample_manifest=tmp_path / "samples.json",
            sample_manifest_sha256="a" * 64,
            semantic_manifest=tmp_path / "semantic.json",
            output=tmp_path / "result.json",
            semantic_device="cpu",
            semantic_batch_size=2,
        )


def test_formal_semantic_manifest_and_model_hashes_are_fail_closed(
    tmp_path: Path,
) -> None:
    bert_dir = tmp_path / "bert"
    mpnet_dir = tmp_path / "mpnet"
    bert_dir.mkdir()
    mpnet_dir.mkdir()
    (bert_dir / "weights.bin").write_bytes(b"bert")
    (mpnet_dir / "weights.bin").write_bytes(b"mpnet")
    manifest = seal_manifest(
        {
            "schema_version": semantic_eval.SEMANTIC_MANIFEST_SCHEMA_VERSION,
            "created_for_evaluation_id": "frozen-test-model-assets",
            "network_at_scoring_time": False,
            "models": {
                "bertscore": {
                    "repo_id": "test/bert",
                    "resolved_revision": "bert-revision",
                    "local_path": str(bert_dir),
                    "directory_sha256": semantic_eval._sha256_path(bert_dir),
                    "num_layers": 17,
                },
                "embedding_cosine": {
                    "repo_id": "test/mpnet",
                    "resolved_revision": "mpnet-revision",
                    "local_path": str(mpnet_dir),
                    "directory_sha256": semantic_eval._sha256_path(mpnet_dir),
                    "independent_from_training_reward": True,
                },
            },
        }
    )
    manifest_path = tmp_path / "semantic.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    bert, mpnet, provenance = subject.load_formal_semantic_backends(
        semantic_manifest_path=manifest_path,
        batch_size=2,
        device="cpu",
    )
    assert isinstance(bert, semantic_eval.BERTScoreBackend)
    assert isinstance(mpnet, semantic_eval.MPNetCosineBackend)
    assert provenance["network_at_scoring_time"] is False

    (mpnet_dir / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(subject.NativeSweepSemanticError, match="directory changed"):
        subject.load_formal_semantic_backends(
            semantic_manifest_path=manifest_path,
            batch_size=2,
            device="cpu",
        )
