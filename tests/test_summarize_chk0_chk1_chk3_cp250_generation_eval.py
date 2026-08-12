from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jobs.eval import summarize_chk0_chk1_chk3_cp250_generation_eval as subject
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    manifest_payload_sha256,
    seal_manifest,
    validate_manifest_integrity,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_sealed(path: Path, value: dict) -> dict:
    result = seal_manifest(value)
    _write_json(path, result)
    return result


def _binding(path: Path, *, sealed: dict | None = None) -> dict:
    result = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    if sealed is not None:
        result["payload_sha256"] = sealed["integrity"]["payload_sha256"]
    return result


def _direction(metric: str) -> str:
    if metric in {
        "repetition_rate",
        "novel_number_rate",
        "rule_covered_unsupported_rate",
    }:
        return "lower"
    if metric in {"generated_token_count", "length_ratio"}:
        return "descriptive"
    return "higher"


def _source_fixture(root: Path, *, delivery_failure: bool) -> dict:
    shared = root / "shared"
    shared.mkdir()
    prompts = shared / "prompts.jsonl"
    references = shared / "references.jsonl"
    prompt_rows = [
        {
            "sample_id": f"sample-{index:02d}",
            "meeting_id": f"meeting-{index // 3:02d}",
        }
        for index in range(subject.EXPECTED_ROWS_PER_ARTIFACT)
    ]
    _write_jsonl(prompts, prompt_rows)
    _write_jsonl(
        references,
        [
            {"sample_id": row["sample_id"], "reference": "official minutes"}
            for row in prompt_rows
        ],
    )
    config = shared / "config.json"
    _write_json(config, {"schema_version": "checkpoint-generation-eval-config-v1"})
    test_manifest_path = shared / "test_manifest.json"
    test_manifest = _write_sealed(
        test_manifest_path,
        {
            "schema_version": "checkpoint-eval-test-manifest-v1",
            "sample_count": 33,
        },
    )
    sealed_source_path = shared / "sealed_source.json"
    sealed_source = _write_sealed(
        sealed_source_path,
        {"schema_version": "fixture-source-v1", "status": "validated"},
    )
    plain_source_path = shared / "plain_source.json"
    _write_json(plain_source_path, {"schema_version": "fixture-plain-v1"})

    generations: dict[str, dict] = {}
    for artifact_number, artifact_id in enumerate(subject.EXPECTED_ARTIFACT_IDS):
        generation_path = shared / f"{artifact_number}.jsonl"
        rows = []
        for index, prompt in enumerate(prompt_rows):
            if delivery_failure:
                generated = f"unfinished raw reasoning {artifact_number} {index}"
                valid = False
                finish_reason = "length"
                response_format = "plain"
                output_tokens = 8192
            else:
                generated = f"reasoning {artifact_number} {index}</think>final text"
                valid = True
                finish_reason = "stop"
                response_format = "deepseek_think_completion"
                output_tokens = 100 + index
            rows.append(
                {
                    "artifact_id": artifact_id,
                    "sample_id": prompt["sample_id"],
                    "meeting_id": prompt["meeting_id"],
                    "generated": generated,
                    "final_answer": "" if delivery_failure else "final text",
                    "valid_generation": valid,
                    "generation_finish_reason": finish_reason,
                    "generation_stop_reason": None if delivery_failure else "eos",
                    "output_token_count": output_tokens,
                    "response_format": response_format,
                }
            )
        _write_jsonl(generation_path, rows)
        manifest_path = shared / f"{artifact_number}.manifest.json"
        manifest = _write_sealed(
            manifest_path,
            {
                "schema_version": "checkpoint-artifact-generation-manifest-v2",
                "artifact_id": artifact_id,
                "sample_count": 33,
                "valid_count": sum(row["valid_generation"] for row in rows),
                "invalid_count": sum(not row["valid_generation"] for row in rows),
                "output": {
                    "path": str(generation_path.resolve()),
                    "sha256": sha256_file(generation_path),
                    "row_count": 33,
                },
            },
        )
        generations[artifact_id] = {
            "generation": _binding(generation_path),
            "manifest": _binding(manifest_path, sealed=manifest),
            "model_sha256": hashlib.sha256(artifact_id.encode()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256((artifact_id + "-tok").encode()).hexdigest(),
            "sample_count": 33,
            "progress": {
                "contract_sha256": "a" * 64,
                "state_sha256": "b" * 64,
                "partial_sha256": "c" * 64,
                "row_count": 33,
            },
        }
    return {
        "frozen_test": {
            "evaluation_id": "three-leg-test",
            "sample_count": 33,
            "meeting_count": 11,
            "prospective_meeting_count": 9,
            "section_count": 3,
            "reference_free": True,
            "prompts": _binding(prompts),
            "references": _binding(references),
            "test_manifest": _binding(test_manifest_path, sealed=test_manifest),
            "evaluation_config": _binding(config),
        },
        "generations": generations,
        "sealed_source": _binding(sealed_source_path, sealed=sealed_source),
        "plain_source": _binding(plain_source_path),
    }


def _make_bundle(
    root: Path,
    *,
    policy_name: str,
    sources: dict,
    delivery_failure: bool,
) -> Path:
    directory = root / policy_name
    directory.mkdir()
    row_scores_path = directory / "row_scores.jsonl"
    claim_checks_path = directory / "claim_checks.jsonl"
    summary_path = directory / "summary.jsonl"
    contrasts_path = directory / "contrasts.jsonl"
    audit_path = directory / "audit.json"
    receipt_path = directory / "input_validation.json"
    policy_id = subject.POLICY_SPECS[policy_name]["policy_id"]

    row_scores: list[dict] = []
    for artifact_id in subject.EXPECTED_ARTIFACT_IDS:
        for index in range(33):
            row_scores.append(
                {
                    "schema_version": subject.EVALUATOR_SCHEMA_VERSION,
                    "artifact_id": artifact_id,
                    "sample_id": f"sample-{index:02d}",
                    "meeting_id": f"meeting-{index // 3:02d}",
                    "scoring_policy": policy_id,
                    "status": (
                        "invalid_output"
                        if delivery_failure and policy_name == "strict"
                        else "scored"
                    ),
                    "valid_output": 0.0 if delivery_failure and policy_name == "strict" else 1.0,
                    "candidate_extraction_mode": (
                        "full_completion" if policy_name == "length_tolerant" else None
                    ),
                }
            )
    _write_jsonl(row_scores_path, row_scores)
    claim_checks = [
        {
            "schema_version": subject.EVALUATOR_SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "sample_id": "sample-00",
        }
        for artifact_id in subject.EXPECTED_ARTIFACT_IDS
    ]
    _write_jsonl(claim_checks_path, claim_checks)

    summary_rows: list[dict] = []
    valid_count = 0 if delivery_failure and policy_name == "strict" else None
    for subset, population in subject.EXPECTED_SUBSET_ROWS.items():
        meetings = 11 if subset == "all_11_meetings" else 9
        for artifact_number, artifact_id in enumerate(subject.EXPECTED_ARTIFACT_IDS):
            artifact_valid = population if valid_count is None else valid_count
            summary_rows.append(
                {
                    "schema_version": subject.EVALUATOR_SCHEMA_VERSION,
                    "artifact_id": artifact_id,
                    "evaluation_subset": subset,
                    "metric": "valid_output",
                    "metric_direction": "higher",
                    "mean": artifact_valid / population,
                    "median": float(bool(artifact_valid)),
                    "ci_lower": 0.0,
                    "ci_upper": 1.0,
                    "n_rows_total": population,
                    "n_rows_eligible": population,
                    "n_rows_ineligible": 0,
                    "n_valid_output": artifact_valid,
                    "n_invalid_output": population - artifact_valid,
                    "n_missing_input": 0,
                    "n_meetings": meetings,
                    "n_meetings_total": meetings,
                }
            )
            for metric in subject.HEADLINE_METRICS:
                mean = 0.2 + artifact_number * 0.1
                summary_rows.append(
                    {
                        "schema_version": subject.EVALUATOR_SCHEMA_VERSION,
                        "artifact_id": artifact_id,
                        "evaluation_subset": subset,
                        "metric": metric,
                        "metric_direction": _direction(metric),
                        "bootstrap_samples": 10_000,
                        "bootstrap_seed": 20_260_729,
                        "bootstrap_unit": "meeting_id",
                        "confidence": 0.95,
                        "mean": mean,
                        "median": mean,
                        "ci_lower": mean - 0.02,
                        "ci_upper": mean + 0.02,
                        "n_rows_total": population,
                        "n_rows_eligible": population,
                        "n_rows_ineligible": 0,
                        "n_meetings": meetings,
                        "n_meetings_total": meetings,
                    }
                )
    _write_jsonl(summary_path, summary_rows)

    contrasts: list[dict] = []
    for baseline, candidate in subject.ADJACENT_CONTRASTS:
        baseline_number = subject.EXPECTED_ARTIFACT_IDS.index(baseline)
        candidate_number = subject.EXPECTED_ARTIFACT_IDS.index(candidate)
        for subset, population in subject.EXPECTED_SUBSET_ROWS.items():
            meetings = 11 if subset == "all_11_meetings" else 9
            for metric in subject.HEADLINE_METRICS:
                baseline_mean = 0.2 + baseline_number * 0.1
                candidate_mean = 0.2 + candidate_number * 0.1
                difference = candidate_mean - baseline_mean
                benefit = -difference if _direction(metric) == "lower" else difference
                contrasts.append(
                    {
                        "schema_version": subject.EVALUATOR_SCHEMA_VERSION,
                        "baseline_artifact_id": baseline,
                        "candidate_artifact_id": candidate,
                        "evaluation_subset": subset,
                        "metric": metric,
                        "metric_direction": _direction(metric),
                        "bootstrap_samples": 10_000,
                        "bootstrap_seed": 20_260_729,
                        "bootstrap_unit": "meeting_id",
                        "confidence": 0.95,
                        "mean_baseline": baseline_mean,
                        "mean_candidate": candidate_mean,
                        "paired_mean_difference": difference,
                        "ci_lower": difference - 0.02,
                        "ci_upper": difference + 0.02,
                        "benefit_difference": benefit,
                        "benefit_ci_lower": benefit - 0.02,
                        "benefit_ci_upper": benefit + 0.02,
                        "n_common_samples_total": population,
                        "n_pairs_eligible": population,
                        "n_pairs_ineligible": 0,
                        "n_meetings": meetings,
                        "n_meetings_total": meetings,
                        "inference_status": "available",
                        "p_value": 0.02,
                        "p_value_holm": 0.04,
                        "effect_size_paired_dz": 0.5,
                    }
                )
    _write_jsonl(contrasts_path, contrasts)

    scoring_policy = {"policy_id": policy_id}
    run_spec = {
        "schema_version": subject.EVALUATOR_SCHEMA_VERSION,
        "artifact_ids": list(subject.EXPECTED_ARTIFACT_IDS),
        "required_artifact_ids": list(subject.EXPECTED_ARTIFACT_IDS),
        "expected_artifact_count": 3,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20_260_729,
        "evaluation_subsets": {
            "all_11_meetings": None,
            "prospective_only_9_meetings": [f"meeting-{index:02d}" for index in range(2, 11)],
        },
        "common_binding": "same-three-leg-run",
        "scoring_policy": scoring_policy,
        "invalid_output_policy_version": policy_name,
    }
    output_paths = {
        "row_scores": (row_scores_path, row_scores),
        "claim_checks": (claim_checks_path, claim_checks),
        "summary": (summary_path, summary_rows),
        "contrasts": (contrasts_path, contrasts),
    }
    audit = _write_sealed(
        audit_path,
        {
            "schema_version": subject.EVALUATOR_SCHEMA_VERSION,
            "status": "validated",
            "immutable": True,
            "artifact_count": 3,
            "artifact_ids": list(subject.EXPECTED_ARTIFACT_IDS),
            "alignment": {
                "complete": True,
                "artifact_ids": list(subject.EXPECTED_ARTIFACT_IDS),
                "sample_universe_count": 33,
                "reference_sample_count": 33,
                "evidence_sample_count": 33,
                "generation_sample_count_by_artifact": {
                    artifact_id: 33 for artifact_id in subject.EXPECTED_ARTIFACT_IDS
                },
                "missing_reference_sample_ids": [],
                "missing_evidence_sample_ids": [],
                "reference_without_evidence_sample_ids": [],
                "evidence_without_reference_sample_ids": [],
                "missing_generation_sample_ids_by_artifact": {
                    artifact_id: [] for artifact_id in subject.EXPECTED_ARTIFACT_IDS
                },
            },
            "scoring_policy": scoring_policy,
            "run_spec": run_spec,
            "run_spec_sha256": manifest_payload_sha256(run_spec),
            "row_counts": {name: len(rows) for name, (_, rows) in output_paths.items()},
            "output_artifacts": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "row_count": len(rows),
                }
                for name, (path, rows) in output_paths.items()
            },
        },
    )
    receipt = _write_sealed(
        receipt_path,
        {
            "schema_version": subject.WRAPPER_SCHEMA_VERSION,
            "status": "validated",
            "complete": True,
            "artifact_ids": list(subject.EXPECTED_ARTIFACT_IDS),
            "evaluation_scope": "standalone chk0 -> chk1 -> direct chk3-cp250 comparison",
            "evaluation_only": True,
            "promotable_to_canonical_dag": False,
            "frozen_test": sources["frozen_test"],
            "pair_evidence_chain": {
                "input_validation_payload_sha256": "e" * 64,
                "checkpoint_manifest": sources["sealed_source"],
                "lineage_manifest": sources["sealed_source"],
                "exact_merge_evidence": sources["sealed_source"],
            },
            "chk3_evidence_chain": {
                "manifest": sources["sealed_source"],
                "exact_merge_evidence": sources["sealed_source"],
                "selection_receipt": sources["plain_source"],
                "authorization": sources["plain_source"],
            },
            "three_leg_lineage": {
                "manifest": sources["sealed_source"],
                "parents": {
                    subject.CHK0_ARTIFACT_ID: None,
                    subject.CHK1_ARTIFACT_ID: subject.CHK0_ARTIFACT_ID,
                    subject.CHK3_ARTIFACT_ID: subject.CHK1_ARTIFACT_ID,
                }
            },
            "semantic_manifest": sources["sealed_source"],
            "generations": sources["generations"],
            "core_evaluator": {
                "module": "jobs.eval.eval_checkpoint_generation",
                "expected_artifact_count": 3,
                "required_artifact_ids": list(subject.EXPECTED_ARTIFACT_IDS),
                "paired_unit": "sample_id",
                "cluster_unit": "meeting_id",
                "multiple_testing": "Holm",
            },
            "scoring_policy": policy_id,
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 20_260_729,
            "core_evaluation": {
                "audit_path": str(audit_path.resolve()),
                "audit_sha256": sha256_file(audit_path),
                "audit_payload_sha256": audit["integrity"]["payload_sha256"],
                "run_spec_sha256": audit["run_spec_sha256"],
                "status": "validated",
            },
        },
    )
    assert validate_manifest_integrity(receipt) == receipt["integrity"]["payload_sha256"]
    return directory


def _fixture(tmp_path: Path, *, delivery_failure: bool = False) -> tuple[Path, Path]:
    sources = _source_fixture(tmp_path, delivery_failure=delivery_failure)
    strict = _make_bundle(
        tmp_path,
        policy_name="strict",
        sources=sources,
        delivery_failure=delivery_failure,
    )
    tolerant = _make_bundle(
        tmp_path,
        policy_name="length_tolerant",
        sources=sources,
        delivery_failure=delivery_failure,
    )
    return strict, tolerant


def _reseal(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("integrity")
    mutate(payload)
    _write_json(path, seal_manifest(payload))


def test_writes_sealed_three_leg_summary_with_adjacent_four_group_contrasts(
    tmp_path: Path,
) -> None:
    strict, tolerant = _fixture(tmp_path)
    output = tmp_path / "result.json"

    result = subject.summarize_chk0_chk1_chk3_cp250_generation_eval(
        strict_dir=strict,
        length_tolerant_dir=tolerant,
        output_file=output,
    )

    assert result["status"] == "validated"
    assert result["matrix"]["total_rows"] == 99
    assert result["overall_classification"]["label"] == "stress_benchmark_comparison_available"
    assert len(result["contrasts"]) == 2
    for edge in result["contrasts"].values():
        primary = edge["strict_primary"]["all_11_meetings"]
        secondary = edge["strict_primary"]["prospective_only_9_meetings"]
        assert primary["role"] == "primary"
        assert secondary["role"] == "secondary"
        assert set(primary["metric_groups"]) == set(subject.METRIC_GROUPS)
        assert "raw-completion" in edge[
            "length_tolerant_raw_completion_diagnostics"
        ]["terminology_guard"]
    chk3 = result["artifacts"][subject.CHK3_ARTIFACT_ID]
    assert chk3["generation_delivery"]["normal_stop_count"] == 33
    assert chk3["generation_delivery"]["native_think_boundary"][
        "exactly_one_with_nonempty_suffix_count"
    ] == 33
    assert chk3["strict_primary"]["all_11_meetings"]["valid_output"]["count"] == 33
    assert "raw_completion" in next(
        key for key in chk3 if key.startswith("length_tolerant")
    )
    assert validate_manifest_integrity(json.loads(output.read_text())) == result[
        "integrity"
    ]["payload_sha256"]
    with pytest.raises(FileExistsError, match="immutable summary"):
        subject.summarize_chk0_chk1_chk3_cp250_generation_eval(
            strict_dir=strict,
            length_tolerant_dir=tolerant,
            output_file=output,
        )


def test_all_strict_delivery_failures_are_inconclusive_not_zero_similarity(
    tmp_path: Path,
) -> None:
    strict, tolerant = _fixture(tmp_path, delivery_failure=True)
    result = subject.summarize_chk0_chk1_chk3_cp250_generation_eval(
        strict_dir=strict,
        length_tolerant_dir=tolerant,
        output_file=tmp_path / "result.json",
    )

    classification = result["overall_classification"]
    assert classification["label"] == (
        "stress_benchmark_inconclusive_all_artifacts_delivery_failure"
    )
    assert "not evidence" in classification["zero_metric_guard"]
    for edge in result["contrasts"].values():
        assert edge["strict_primary"]["all_11_meetings"]["classification"] == (
            "stress_benchmark_inconclusive_bilateral_delivery_failure"
        )
    assert result["artifacts"][subject.CHK0_ARTIFACT_ID]["generation_delivery"][
        "output_token_count"
    ]["max"] == 8192


def test_rejects_claim_checks_tampering_after_audit(tmp_path: Path) -> None:
    strict, tolerant = _fixture(tmp_path)
    with (strict / "claim_checks.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="claim_checks hash differs"):
        subject.summarize_chk0_chk1_chk3_cp250_generation_eval(
            strict_dir=strict,
            length_tolerant_dir=tolerant,
            output_file=tmp_path / "result.json",
        )


def test_rejects_resealed_input_validation_with_wrong_audit_hash(tmp_path: Path) -> None:
    strict, tolerant = _fixture(tmp_path)
    _reseal(
        strict / "input_validation.json",
        lambda payload: payload["core_evaluation"].update({"audit_sha256": "f" * 64}),
    )
    with pytest.raises(ValueError, match="core-audit binding changed"):
        subject.summarize_chk0_chk1_chk3_cp250_generation_eval(
            strict_dir=strict,
            length_tolerant_dir=tolerant,
            output_file=tmp_path / "result.json",
        )


def test_rejects_policies_bound_to_different_generation_sources(tmp_path: Path) -> None:
    strict, tolerant = _fixture(tmp_path)
    receipt_path = tolerant / "input_validation.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifact = subject.CHK3_ARTIFACT_ID
    original = Path(receipt["generations"][artifact]["generation"]["path"])
    alternate = tmp_path / "alternate.jsonl"
    alternate.write_bytes(original.read_bytes())
    original_manifest = Path(receipt["generations"][artifact]["manifest"]["path"])
    manifest_payload = json.loads(original_manifest.read_text(encoding="utf-8"))
    manifest_payload.pop("integrity")
    manifest_payload["output"].update(
        {
            "path": str(alternate.resolve()),
            "sha256": sha256_file(alternate),
        }
    )
    alternate_manifest = tmp_path / "alternate.manifest.json"
    alternate_manifest_payload = _write_sealed(alternate_manifest, manifest_payload)
    # Byte-identical alternate paths are still a different sealed source identity.
    _reseal(
        receipt_path,
        lambda payload: (
            payload["generations"][artifact]["generation"].update(
                {"path": str(alternate.resolve()), "sha256": sha256_file(alternate)}
            ),
            payload["generations"][artifact]["manifest"].update(
                _binding(alternate_manifest, sealed=alternate_manifest_payload)
            ),
        ),
    )
    with pytest.raises(ValueError, match="do not bind the same sources"):
        subject.summarize_chk0_chk1_chk3_cp250_generation_eval(
            strict_dir=strict,
            length_tolerant_dir=tolerant,
            output_file=tmp_path / "result.json",
        )


def test_rejects_non_adjacent_contrast_even_when_audit_is_resealed(tmp_path: Path) -> None:
    strict, tolerant = _fixture(tmp_path)
    contrasts_path = strict / "contrasts.jsonl"
    rows = [json.loads(line) for line in contrasts_path.read_text().splitlines()]
    rows[0]["baseline_artifact_id"] = subject.CHK0_ARTIFACT_ID
    rows[0]["candidate_artifact_id"] = subject.CHK3_ARTIFACT_ID
    _write_jsonl(contrasts_path, rows)
    audit_path = strict / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit.pop("integrity")
    audit["row_counts"]["contrasts"] = len(rows)
    audit["output_artifacts"]["contrasts"].update(
        {"sha256": sha256_file(contrasts_path), "row_count": len(rows)}
    )
    audit = seal_manifest(audit)
    _write_json(audit_path, audit)
    receipt_path = strict / "input_validation.json"
    _reseal(
        receipt_path,
        lambda payload: payload["core_evaluation"].update(
            {
                "audit_sha256": sha256_file(audit_path),
                "audit_payload_sha256": audit["integrity"]["payload_sha256"],
            }
        ),
    )
    with pytest.raises(ValueError, match="invalid edge"):
        subject.summarize_chk0_chk1_chk3_cp250_generation_eval(
            strict_dir=strict,
            length_tolerant_dir=tolerant,
            output_file=tmp_path / "result.json",
        )
