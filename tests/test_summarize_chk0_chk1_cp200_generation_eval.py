import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jobs.eval.summarize_chk0_chk1_cp200_generation_eval import (
    BASELINE_ARTIFACT_ID,
    CANDIDATE_ARTIFACT_ID,
    EVALUATOR_SCHEMA_VERSION,
    EXPECTED_ARTIFACT_IDS,
    EXPECTED_SUBSET_ROWS,
    HEADLINE_METRICS,
    METRIC_GROUPS,
    SCHEMA_VERSION,
    WRAPPER_SCHEMA_VERSION,
    summarize_chk0_chk1_cp200_generation_eval,
)
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    manifest_payload_sha256,
    seal_manifest,
    validate_manifest_integrity,
)


PROSPECTIVE_MEETINGS = [
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-10",
    "2026-01-28",
    "2026-03-18",
    "2026-04-29",
    "2026-06-17",
]
PROGRESS_VALIDATION = {
    "contract_sha256": "a" * 64,
    "state_sha256": "b" * 64,
    "partial_sha256": "c" * 64,
    "row_count": 33,
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _direction(metric: str) -> str:
    if metric in {
        "repetition_rate",
        "novel_number_rate",
        "rule_covered_unsupported_rate",
    }:
        return "lower"
    if metric in {"length_ratio", "generated_token_count"}:
        return "descriptive"
    return "higher"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _file_binding(path: Path, *, sealed: bool = False) -> dict:
    binding = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    if sealed:
        payload = json.loads(path.read_text(encoding="utf-8"))
        binding["payload_sha256"] = validate_manifest_integrity(payload)
    return binding


def _ensure_input_sources(root: Path) -> dict:
    sources = root / "input_sources"
    sources.mkdir(exist_ok=True)
    prompts = sources / "prompts.jsonl"
    references = sources / "references.jsonl"
    config = sources / "config.json"
    if not prompts.exists():
        _write_jsonl(
            prompts,
            [{"sample_id": f"sample-{index:02d}"} for index in range(33)],
        )
        _write_jsonl(
            references,
            [{"sample_id": f"sample-{index:02d}"} for index in range(33)],
        )
        _write_json(
            config,
            {"schema_version": "checkpoint-generation-eval-config-v1"},
        )

    sealed_specs = {
        "test_manifest": "checkpoint-eval-test-manifest-v1",
        "checkpoint_manifest": "retrain-v2-checkpoint-eval-manifest-v1",
        "lineage_manifest": "retrain-v2-eval-lineage-v1",
        "semantic_manifest": "checkpoint-eval-semantic-model-manifest-v1",
        "exact_merge_evidence": "lora-merge-lineage-evidence-v1",
    }
    sealed_paths = {}
    for name, schema in sealed_specs.items():
        path = sources / f"{name}.json"
        if not path.exists():
            _write_json(path, seal_manifest({"schema_version": schema, "name": name}))
        sealed_paths[name] = path

    generation_receipts = {}
    for artifact_id in EXPECTED_ARTIFACT_IDS:
        generation_path = sources / f"{artifact_id}.jsonl"
        manifest_path = sources / f"{artifact_id}.manifest.json"
        if not generation_path.exists():
            _write_jsonl(
                generation_path,
                [
                    {
                        "artifact_id": artifact_id,
                        "sample_id": f"sample-{index:02d}",
                    }
                    for index in range(33)
                ],
            )
            _write_json(
                manifest_path,
                seal_manifest(
                    {
                        "schema_version": "checkpoint-artifact-generation-manifest-v2",
                        "artifact_id": artifact_id,
                    }
                ),
            )
        generation_receipts[artifact_id] = {
            "generation": _file_binding(generation_path),
            "manifest": _file_binding(manifest_path, sealed=True),
            "model_sha256": "1" * 64
            if artifact_id == BASELINE_ARTIFACT_ID
            else "2" * 64,
            "tokenizer_sha256": "3" * 64
            if artifact_id == BASELINE_ARTIFACT_ID
            else "4" * 64,
            "sample_count": 33,
            "progress": PROGRESS_VALIDATION,
        }

    return {
        "artifact_ids": list(EXPECTED_ARTIFACT_IDS),
        "frozen_test": {
            "evaluation_id": "fixture-evaluation",
            "sample_count": 33,
            "meeting_count": 11,
            "prospective_meeting_count": 9,
            "section_count": 3,
            "reference_free": True,
            "prompts": _file_binding(prompts),
            "references": _file_binding(references),
            "test_manifest": _file_binding(sealed_paths["test_manifest"], sealed=True),
            "evaluation_config": _file_binding(config),
        },
        "checkpoint_manifest": _file_binding(
            sealed_paths["checkpoint_manifest"], sealed=True
        ),
        "lineage_manifest": _file_binding(
            sealed_paths["lineage_manifest"], sealed=True
        ),
        "semantic_manifest": {
            **_file_binding(sealed_paths["semantic_manifest"], sealed=True),
            "models": {
                "bertscore": {"directory_sha256": "5" * 64},
                "embedding_cosine": {"directory_sha256": "6" * 64},
            },
        },
        "exact_merge_evidence": {
            **_file_binding(sealed_paths["exact_merge_evidence"], sealed=True),
            "subject_artifact_id": "chk1-clean-v2-lr1e6-cp200",
            "conclusion": "exact_base_plus_adapter_merge_verified",
            "algorithm_version": "peft-lora-fp32-exact-v1",
            "sources": {
                "base_model": "base",
                "adapter": "adapter",
                "merged_model": "merged",
            },
            "tensor_verification": {"complete": True, "mismatch_count": 0},
        },
        "generations": generation_receipts,
        "core_evaluator": {
            "module": "jobs.eval.eval_checkpoint_generation",
            "expected_artifact_count": 2,
            "paired_unit": "sample_id",
            "cluster_unit": "meeting_id",
            "multiple_testing": "Holm",
        },
    }


def _rewrite_sealed(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("integrity")
    mutate(payload)
    _write_json(path, seal_manifest(payload))


class TestChk0Chk1Cp200GenerationSummary(unittest.TestCase):
    def setUp(self) -> None:
        self.progress_patcher = patch(
            "jobs.eval.summarize_chk0_chk1_cp200_generation_eval."
            "validate_generation_progress_binding",
            return_value=PROGRESS_VALIDATION,
        )
        self.progress_patcher.start()

    def tearDown(self) -> None:
        self.progress_patcher.stop()

    def _make_bundle(
        self,
        root: Path,
        *,
        policy_name: str,
        common_binding: str = "same-inputs",
        duplicate_matrix_key: bool = False,
        audit_artifact_ids: list[str] | None = None,
    ) -> Path:
        input_sources = _ensure_input_sources(root)
        directory = root / policy_name
        directory.mkdir()
        row_scores_path = directory / "row_scores.jsonl"
        claim_checks_path = directory / "claim_checks.jsonl"
        summary_path = directory / "summary.jsonl"
        contrasts_path = directory / "contrasts.jsonl"
        audit_path = directory / "audit.json"
        receipt_path = directory / "input_validation.json"

        row_scores = []
        for artifact_id in EXPECTED_ARTIFACT_IDS:
            for index in range(33):
                sample_index = (
                    0
                    if duplicate_matrix_key
                    and artifact_id == CANDIDATE_ARTIFACT_ID
                    and index == 1
                    else index
                )
                row_scores.append(
                    {
                        "schema_version": EVALUATOR_SCHEMA_VERSION,
                        "artifact_id": artifact_id,
                        "sample_id": f"sample-{sample_index:02d}",
                    }
                )
        _write_jsonl(row_scores_path, row_scores)
        claim_checks = [
            {
                "schema_version": EVALUATOR_SCHEMA_VERSION,
                "artifact_id": artifact_id,
                "sample_id": "sample-00",
                "claim_id": f"claim-{index}",
            }
            for index, artifact_id in enumerate(EXPECTED_ARTIFACT_IDS)
        ]
        _write_jsonl(claim_checks_path, claim_checks)

        summary_rows = []
        for subset, n_rows in EXPECTED_SUBSET_ROWS.items():
            n_meetings = 11 if subset == "all_11_meetings" else 9
            for artifact_id in EXPECTED_ARTIFACT_IDS:
                invalid = 1 if artifact_id == BASELINE_ARTIFACT_ID else 2
                summary_rows.append(
                    {
                        "schema_version": EVALUATOR_SCHEMA_VERSION,
                        "artifact_id": artifact_id,
                        "evaluation_subset": subset,
                        "metric": "valid_output",
                        "metric_direction": "higher",
                        "bootstrap_samples": 10_000,
                        "bootstrap_seed": 20260729,
                        "bootstrap_unit": "meeting_id",
                        "confidence": 0.95,
                        "mean": (n_rows - invalid) / n_rows,
                        "median": 1.0,
                        "ci_lower": 0.8,
                        "ci_upper": 1.0,
                        "n_rows_total": n_rows,
                        "n_rows_eligible": n_rows,
                        "n_rows_ineligible": 0,
                        "n_valid_output": n_rows - invalid,
                        "n_invalid_output": invalid,
                        "n_missing_input": 0,
                        "n_meetings": n_meetings,
                        "n_meetings_total": n_meetings,
                    }
                )
                for metric in HEADLINE_METRICS:
                    direction = _direction(metric)
                    base_mean = 1.0 if metric == "length_ratio" else 0.4
                    delta = -0.1 if direction == "lower" else 0.1
                    mean = (
                        base_mean
                        if artifact_id == BASELINE_ARTIFACT_ID
                        else base_mean + delta
                    )
                    summary_rows.append(
                        {
                            "schema_version": EVALUATOR_SCHEMA_VERSION,
                            "artifact_id": artifact_id,
                            "evaluation_subset": subset,
                            "metric": metric,
                            "metric_direction": direction,
                            "bootstrap_samples": 10_000,
                            "bootstrap_seed": 20260729,
                            "bootstrap_unit": "meeting_id",
                            "confidence": 0.95,
                            "mean": mean,
                            "median": mean,
                            "ci_lower": mean - 0.05,
                            "ci_upper": mean + 0.05,
                            "n_rows_total": n_rows,
                            "n_rows_eligible": n_rows,
                            "n_rows_ineligible": 0,
                            "n_valid_output": n_rows - invalid,
                            "n_invalid_output": invalid,
                            "n_missing_input": 0,
                            "n_meetings": n_meetings,
                            "n_meetings_total": n_meetings,
                        }
                    )
        _write_jsonl(summary_path, summary_rows)

        contrast_rows = []
        for subset, n_rows in EXPECTED_SUBSET_ROWS.items():
            n_meetings = 11 if subset == "all_11_meetings" else 9
            for metric in HEADLINE_METRICS:
                direction = _direction(metric)
                baseline_mean = 1.0 if metric == "length_ratio" else 0.4
                difference = -0.1 if direction == "lower" else 0.1
                benefit = -difference if direction == "lower" else difference
                contrast_rows.append(
                    {
                        "schema_version": EVALUATOR_SCHEMA_VERSION,
                        "baseline_artifact_id": BASELINE_ARTIFACT_ID,
                        "candidate_artifact_id": CANDIDATE_ARTIFACT_ID,
                        "evaluation_subset": subset,
                        "metric": metric,
                        "metric_direction": direction,
                        "bootstrap_samples": 10_000,
                        "bootstrap_seed": 20260729,
                        "bootstrap_unit": "meeting_id",
                        "confidence": 0.95,
                        "mean_baseline": baseline_mean,
                        "mean_candidate": baseline_mean + difference,
                        "paired_mean_difference": difference,
                        "ci_lower": difference - 0.02,
                        "ci_upper": difference + 0.02,
                        "benefit_difference": benefit,
                        "benefit_ci_lower": benefit - 0.02,
                        "benefit_ci_upper": benefit + 0.02,
                        "n_common_samples_total": n_rows,
                        "n_pairs_eligible": n_rows,
                        "n_pairs_ineligible": 0,
                        "n_meetings": n_meetings,
                        "n_meetings_total": n_meetings,
                        "inference_status": "available",
                        "p_value": 0.02,
                        "p_value_holm": 0.04,
                        "effect_size_paired_dz": 0.5,
                    }
                )
        _write_jsonl(contrasts_path, contrast_rows)

        policy_id = (
            "strict-final-answer-v2"
            if policy_name == "strict"
            else "length-tolerant-open-tags-v1"
        )
        scoring_policy = {"policy_id": policy_id}
        run_spec = {
            "schema_version": EVALUATOR_SCHEMA_VERSION,
            "artifact_ids": list(EXPECTED_ARTIFACT_IDS),
            "required_artifact_ids": list(EXPECTED_ARTIFACT_IDS),
            "expected_artifact_count": 2,
            "evaluation_subsets": {
                "all_11_meetings": None,
                "prospective_only_9_meetings": PROSPECTIVE_MEETINGS,
            },
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 20260729,
            "common_binding": common_binding,
            "scoring_policy": scoring_policy,
            "invalid_output_policy_version": policy_name,
        }
        artifact_ids = audit_artifact_ids or list(EXPECTED_ARTIFACT_IDS)
        audit = seal_manifest(
            {
                "schema_version": EVALUATOR_SCHEMA_VERSION,
                "status": "validated",
                "immutable": True,
                "artifact_count": 2,
                "artifact_ids": artifact_ids,
                "alignment": {
                    "complete": True,
                    "artifact_ids": list(EXPECTED_ARTIFACT_IDS),
                    "sample_universe_count": 33,
                    "reference_sample_count": 33,
                    "evidence_sample_count": 33,
                    "generation_sample_count_by_artifact": {
                        artifact_id: 33 for artifact_id in EXPECTED_ARTIFACT_IDS
                    },
                    "missing_evidence_sample_ids": [],
                    "missing_reference_sample_ids": [],
                    "reference_without_evidence_sample_ids": [],
                    "evidence_without_reference_sample_ids": [],
                    "missing_generation_sample_ids_by_artifact": {
                        artifact_id: [] for artifact_id in EXPECTED_ARTIFACT_IDS
                    },
                },
                "scoring_policy": scoring_policy,
                "run_spec": run_spec,
                "run_spec_sha256": manifest_payload_sha256(run_spec),
                "row_counts": {
                    "row_scores": len(row_scores),
                    "claim_checks": len(claim_checks),
                    "summary": len(summary_rows),
                    "contrasts": len(contrast_rows),
                },
                "output_artifacts": {
                    "row_scores": {
                        "path": str(row_scores_path.resolve()),
                        "sha256": sha256_file(row_scores_path),
                        "row_count": len(row_scores),
                    },
                    "claim_checks": {
                        "path": str(claim_checks_path.resolve()),
                        "sha256": sha256_file(claim_checks_path),
                        "row_count": len(claim_checks),
                    },
                    "summary": {
                        "path": str(summary_path.resolve()),
                        "sha256": sha256_file(summary_path),
                        "row_count": len(summary_rows),
                    },
                    "contrasts": {
                        "path": str(contrasts_path.resolve()),
                        "sha256": sha256_file(contrasts_path),
                        "row_count": len(contrast_rows),
                    },
                },
            }
        )
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        audit_payload_sha256 = validate_manifest_integrity(audit)
        receipt = seal_manifest(
            {
                "schema_version": WRAPPER_SCHEMA_VERSION,
                "status": "validated",
                "complete": True,
                **input_sources,
                "scoring_policy": policy_id,
                "bootstrap_samples": 10_000,
                "bootstrap_seed": 20260729,
                "core_evaluation": {
                    "audit_path": str(audit_path.resolve()),
                    "audit_sha256": sha256_file(audit_path),
                    "audit_payload_sha256": audit_payload_sha256,
                    "run_spec_sha256": audit["run_spec_sha256"],
                    "status": "validated",
                },
            }
        )
        _write_json(receipt_path, receipt)
        return directory

    def test_writes_one_sealed_four_group_summary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            strict = self._make_bundle(root, policy_name="strict")
            tolerant = self._make_bundle(root, policy_name="length_tolerant")
            output = root / "result.json"

            result = summarize_chk0_chk1_cp200_generation_eval(
                strict_dir=strict,
                length_tolerant_dir=tolerant,
                output_file=output,
            )

            self.assertEqual(result["schema_version"], SCHEMA_VERSION)
            self.assertEqual(result["comparison"]["matrix"]["total_rows"], 66)
            self.assertEqual(
                set(
                    result["policies"]["strict"]["subsets"]["all_11_meetings"][
                        "metric_groups"
                    ]
                ),
                set(METRIC_GROUPS),
            )
            rouge = result["policies"]["strict"]["subsets"]["all_11_meetings"][
                "metric_groups"
            ]["surface"]["rouge_l_f1"]
            self.assertAlmostEqual(rouge["paired_difference"]["estimate"], 0.1)
            self.assertEqual(rouge["paired_difference"]["n_pairs_eligible"], 33)
            self.assertEqual(
                result["policies"]["strict"]["subsets"]["all_11_meetings"][
                    "invalid_counts"
                ][CANDIDATE_ARTIFACT_ID]["n_invalid_output"],
                2,
            )
            self.assertEqual(
                result["policies"]["length_tolerant"]["analysis_role"],
                "diagnostic_robustness_only",
            )
            strict_policy = result["policies"]["strict"]
            self.assertIn(
                "bertscore_precision",
                strict_policy["subsets"]["all_11_meetings"]["metric_groups"][
                    "semantic"
                ],
            )
            self.assertIn(
                "generated_token_count",
                strict_policy["subsets"]["all_11_meetings"]["metric_groups"]["surface"],
            )
            self.assertEqual(
                strict_policy["source_artifacts"]["claim_checks"]["row_count"], 2
            )
            self.assertIn(
                "payload_sha256", strict_policy["source_artifacts"]["input_validation"]
            )
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                validate_manifest_integrity(written),
                result["integrity"]["payload_sha256"],
            )
            with self.assertRaisesRegex(FileExistsError, "immutable summary"):
                summarize_chk0_chk1_cp200_generation_eval(
                    strict_dir=strict,
                    length_tolerant_dir=tolerant,
                    output_file=output,
                )

    def test_rejects_claim_checks_changed_after_audit_was_sealed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            strict = self._make_bundle(root, policy_name="strict")
            tolerant = self._make_bundle(root, policy_name="length_tolerant")
            with (strict / "claim_checks.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "claim_checks hash differs"):
                summarize_chk0_chk1_cp200_generation_eval(
                    strict_dir=strict,
                    length_tolerant_dir=tolerant,
                    output_file=root / "result.json",
                )

    def test_rejects_input_validation_changed_after_sealing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            strict = self._make_bundle(root, policy_name="strict")
            tolerant = self._make_bundle(root, policy_name="length_tolerant")
            receipt_path = strict / "input_validation.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["complete"] = False
            _write_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "payload digest mismatch"):
                summarize_chk0_chk1_cp200_generation_eval(
                    strict_dir=strict,
                    length_tolerant_dir=tolerant,
                    output_file=root / "result.json",
                )

    def test_rejects_resealed_receipt_with_wrong_core_audit_binding(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            strict = self._make_bundle(root, policy_name="strict")
            tolerant = self._make_bundle(root, policy_name="length_tolerant")
            _rewrite_sealed(
                strict / "input_validation.json",
                lambda payload: payload["core_evaluation"].update(
                    {"audit_sha256": "f" * 64}
                ),
            )
            with self.assertRaisesRegex(ValueError, "core-audit binding changed"):
                summarize_chk0_chk1_cp200_generation_eval(
                    strict_dir=strict,
                    length_tolerant_dir=tolerant,
                    output_file=root / "result.json",
                )

    def test_rejects_two_policies_bound_to_different_frozen_sources(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            strict = self._make_bundle(root, policy_name="strict")
            tolerant = self._make_bundle(root, policy_name="length_tolerant")
            alternate = root / "alternate_config.json"
            _write_json(
                alternate,
                {
                    "schema_version": "checkpoint-generation-eval-config-v1",
                    "different": True,
                },
            )
            _rewrite_sealed(
                tolerant / "input_validation.json",
                lambda payload: payload["frozen_test"].update(
                    {"evaluation_config": _file_binding(alternate)}
                ),
            )
            with self.assertRaisesRegex(ValueError, "same frozen inputs"):
                summarize_chk0_chk1_cp200_generation_eval(
                    strict_dir=strict,
                    length_tolerant_dir=tolerant,
                    output_file=root / "result.json",
                )

    def test_rejects_summary_changed_after_audit_was_sealed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            strict = self._make_bundle(root, policy_name="strict")
            tolerant = self._make_bundle(root, policy_name="length_tolerant")
            with (strict / "summary.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "summary hash differs"):
                summarize_chk0_chk1_cp200_generation_eval(
                    strict_dir=strict,
                    length_tolerant_dir=tolerant,
                    output_file=root / "result.json",
                )

    def test_rejects_audit_changed_after_it_was_sealed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            strict = self._make_bundle(root, policy_name="strict")
            tolerant = self._make_bundle(root, policy_name="length_tolerant")
            audit_path = strict / "audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["artifact_count"] = 3
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "payload digest mismatch"):
                summarize_chk0_chk1_cp200_generation_eval(
                    strict_dir=strict,
                    length_tolerant_dir=tolerant,
                    output_file=root / "result.json",
                )

    def test_rejects_duplicate_key_in_actual_2_by_33_matrix(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            strict = self._make_bundle(
                root, policy_name="strict", duplicate_matrix_key=True
            )
            tolerant = self._make_bundle(root, policy_name="length_tolerant")
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                summarize_chk0_chk1_cp200_generation_eval(
                    strict_dir=strict,
                    length_tolerant_dir=tolerant,
                    output_file=root / "result.json",
                )

    def test_rejects_wrong_artifact_inventory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            strict = self._make_bundle(
                root,
                policy_name="strict",
                audit_artifact_ids=[BASELINE_ARTIFACT_ID, "wrong-candidate"],
            )
            tolerant = self._make_bundle(root, policy_name="length_tolerant")
            with self.assertRaisesRegex(ValueError, "expected chk0/chk1 artifact IDs"):
                summarize_chk0_chk1_cp200_generation_eval(
                    strict_dir=strict,
                    length_tolerant_dir=tolerant,
                    output_file=root / "result.json",
                )

    def test_rejects_strict_and_tolerant_from_different_runs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            strict = self._make_bundle(
                root, policy_name="strict", common_binding="run-a"
            )
            tolerant = self._make_bundle(
                root, policy_name="length_tolerant", common_binding="run-b"
            )
            with self.assertRaisesRegex(ValueError, "not bound to the same run inputs"):
                summarize_chk0_chk1_cp200_generation_eval(
                    strict_dir=strict,
                    length_tolerant_dir=tolerant,
                    output_file=root / "result.json",
                )


if __name__ == "__main__":
    unittest.main()
