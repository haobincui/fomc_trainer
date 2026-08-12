import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jobs.generation.finalize_loo_scoped_workflow import (
    _require_matching_scoring_protocols,
    _validate_results_manifest,
)
from open_r1.provenance import sha256_file


class TestScopedWorkflowScoringProtocol(unittest.TestCase):
    def _results_fixture(
        self,
        root: Path,
        *,
        population_id: str,
        embedding_max_tokens: int = 4096,
        embedding_path: str = "/models/frozen-embedding",
        source_text: str = "frozen Actual Minutes source\n",
    ) -> Path:
        population_root = root / population_id
        population_root.mkdir(parents=True)
        source_path = population_root / "actual_minutes_source.jsonl"
        source_path.write_text(source_text, encoding="utf-8")
        reference_manifest_path = population_root / "reference_manifest.json"
        reference_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "loo-actual-minutes-reference-view-v1",
                    "source": {
                        "path": str(source_path.resolve()),
                        "sha256": sha256_file(source_path),
                        "text_field": "reference",
                    },
                }
            ),
            encoding="utf-8",
        )
        inputs = {}
        for arm in (
            "deletion_primary",
            "neutral_primary",
            "deletion_stochastic",
            "neutral_stochastic",
        ):
            audit_path = population_root / f"{arm}.audit.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "schema_version": "loo-paired-v3",
                        "target_mode": "actual-minutes",
                        "primary_estimand": (
                            "delta = similarity_full - similarity_masked"
                        ),
                        "embedding_model_artifact": {
                            "path": embedding_path,
                            "sha256": "a" * 64,
                            "kind": "directory",
                        },
                        "embedding_batch_size": 1,
                        "embedding_max_tokens": embedding_max_tokens,
                        "embedding_long_text_policy": "chunk-mean",
                        "score_chunk_size": 64,
                        "reference_key_fields": [
                            "meeting_date",
                            "section_name",
                        ],
                        "reference_text_field": "response",
                        "reference_duplicate_policy": "error",
                        "scoped_experiment": {
                            "reference_manifest": {
                                "path": str(reference_manifest_path.resolve()),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            inputs[arm] = {"audit_file": str(audit_path.resolve())}
        report_path = population_root / "results_manifest.json"
        report_path.write_text(
            json.dumps(
                {
                    "inputs": inputs,
                    "estimand": (
                        "delta = cos(full_output, actual_minutes) - "
                        "cos(intervention_output, actual_minutes)"
                    ),
                    "aggregation_order": [
                        "equal-weight sections within meeting-replicate",
                        "equal-weight replicates within meeting",
                        "equal-weight meetings",
                    ],
                    "inference_unit": "meeting",
                    "bootstrap": {
                        "kind": "joint_meeting_cluster_percentile",
                        "draws": 10000,
                        "seed": 20260728,
                        "confidence": 0.95,
                    },
                    "multiplicity": {
                        "method": "Holm",
                        "family_size": 6,
                    },
                }
            ),
            encoding="utf-8",
        )
        return report_path

    @patch(
        "jobs.generation.finalize_loo_scoped_workflow.validate_report_manifest"
    )
    def test_freezes_and_compares_complete_scoring_protocol(
        self,
        mock_validate_report_manifest,
    ):
        mock_validate_report_manifest.return_value = {"status": "validated"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scope_path = root / "scope.json"
            scope_path.write_text("{}\n", encoding="utf-8")
            pilot_path = self._results_fixture(
                root,
                population_id="pilot_eval_13",
            )
            formal_path = self._results_fixture(
                root,
                population_id="formal_test_13",
            )
            pilot = _validate_results_manifest(
                pilot_path,
                scope_path=scope_path,
                population_id="pilot_eval_13",
            )
            formal = _validate_results_manifest(
                formal_path,
                scope_path=scope_path,
                population_id="formal_test_13",
            )
            _require_matching_scoring_protocols(pilot, formal)
            self.assertEqual(pilot["embedding"]["max_tokens"], 4096)
            self.assertEqual(
                pilot["embedding"]["artifact_path"],
                "/models/frozen-embedding",
            )
            self.assertEqual(pilot["embedding"]["long_text_policy"], "chunk-mean")
            self.assertEqual(pilot["reference"]["source_text_field"], "reference")

            changed_formal_path = self._results_fixture(
                root,
                population_id="formal_changed",
                embedding_max_tokens=2048,
            )
            changed = _validate_results_manifest(
                changed_formal_path,
                scope_path=scope_path,
                population_id="formal_test_13",
            )
            with self.assertRaisesRegex(ValueError, "scoring protocols differ"):
                _require_matching_scoring_protocols(pilot, changed)

            changed_path = self._results_fixture(
                root,
                population_id="formal_path_changed",
                embedding_path="/models/other-embedding-copy",
            )
            path_signature = _validate_results_manifest(
                changed_path,
                scope_path=scope_path,
                population_id="formal_test_13",
            )
            with self.assertRaisesRegex(ValueError, "scoring protocols differ"):
                _require_matching_scoring_protocols(pilot, path_signature)

            changed_source_path = self._results_fixture(
                root,
                population_id="formal_source_changed",
                source_text="different Actual Minutes source\n",
            )
            changed_source = _validate_results_manifest(
                changed_source_path,
                scope_path=scope_path,
                population_id="formal_test_13",
            )
            with self.assertRaisesRegex(ValueError, "scoring protocols differ"):
                _require_matching_scoring_protocols(pilot, changed_source)

    @patch(
        "jobs.generation.finalize_loo_scoped_workflow.validate_report_manifest"
    )
    def test_rejects_changed_underlying_reference_source(
        self,
        mock_validate_report_manifest,
    ):
        mock_validate_report_manifest.return_value = {"status": "validated"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scope_path = root / "scope.json"
            scope_path.write_text("{}\n", encoding="utf-8")
            report_path = self._results_fixture(
                root,
                population_id="pilot_eval_13",
            )
            source_path = root / "pilot_eval_13/actual_minutes_source.jsonl"
            source_path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "Actual-Minutes source changed or is missing",
            ):
                _validate_results_manifest(
                    report_path,
                    scope_path=scope_path,
                    population_id="pilot_eval_13",
                )


if __name__ == "__main__":
    unittest.main()
