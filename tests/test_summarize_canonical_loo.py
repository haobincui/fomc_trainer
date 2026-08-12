import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from jobs.eval.eval_leave_one_out import (
    DELETION_STRATEGY,
    NEUTRAL_STRATEGY,
    _load_scoped_experiment,
)
from jobs.eval.summarize_canonical_loo import (
    ARM_SPECS,
    _holm_six_deletion_family,
    _joint_bootstrap_metrics,
    _t_test,
    aggregate_sections_replicates_meetings,
    summarize_canonical_loo,
    validate_report_manifest,
)
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import seal_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPE_FILE = REPO_ROOT / "configs/main/loo_experiment_legacy6.json"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class TestCanonicalLooSummary(unittest.TestCase):
    @staticmethod
    def _mock_release_validation(**kwargs) -> dict:
        release_path = Path(kwargs["release_manifest_file"]).resolve()
        release = json.loads(release_path.read_text(encoding="utf-8"))
        strategy = kwargs["intervention_strategy"]
        prefix = "deletion" if strategy == DELETION_STRATEGY else "neutral"
        return {
            "status": "validated",
            "path": str(release_path),
            "sha256": sha256_file(release_path),
            "payload_sha256": release["integrity"]["payload_sha256"],
            "release_kind": "pilot",
            "run_id": "fixture-run",
            "arm": f"{prefix}_{kwargs['regime']}",
            "arm_generation_manifest_sha256": sha256_file(
                Path(kwargs["input_folder"]) / "generation_manifest.json"
            ),
            "minutes_model_sha256": "3" * 64,
            "minutes_tokenizer_sha256": "4" * 64,
        }

    def _dependency_fixture(self, root: Path) -> dict:
        scope = _load_scoped_experiment(SCOPE_FILE)
        reference_path = root / "actual_minutes_reference.jsonl"
        reference_manifest_path = root / "actual_minutes_reference_manifest.json"
        release_path = root / "scoped_generation_release_manifest.json"
        generation_paths = {
            arm: root / "generations" / arm / "generation_manifest.json"
            for arm in ARM_SPECS
        }
        if not release_path.exists():
            reference_rows = [
                {
                    "meeting_date": meeting_date,
                    "section_name": section_name,
                    "response": f"reference {meeting_date} {section_name}",
                }
                for meeting_date in scope["populations"]["pilot_eval_13"][
                    "meeting_dates"
                ]
                for section_name in scope["section_names"]
            ]
            _write_jsonl(reference_path, reference_rows)
            reference_manifest = {
                "schema_version": "loo-actual-minutes-reference-view-v1",
                "status": "complete",
                "experiment_id": scope["experiment_id"],
                "population_id": "pilot_eval_13",
                "row_count": 39,
                "scope_manifest": {
                    "path": scope["_path"],
                    "sha256": scope["_sha256"],
                },
                "output": {
                    "path": str(reference_path.resolve()),
                    "sha256": sha256_file(reference_path),
                    "row_count": 39,
                    "key_fields": ["meeting_date", "section_name"],
                    "text_field": "response",
                },
            }
            reference_manifest_path.write_text(
                json.dumps(reference_manifest), encoding="utf-8"
            )
            for arm, path in generation_paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "arm": arm,
                            "model_artifact": {"sha256": "3" * 64},
                            "tokenizer_artifact": {"sha256": "4" * 64},
                        }
                    ),
                    encoding="utf-8",
                )
            release = seal_manifest(
                {
                    "schema_version": "loo-scoped-generation-release-v1",
                    "status": "complete",
                    "release_kind": "pilot",
                    "scoped_population_release": True,
                    "standalone_full_roster_canonical_release": False,
                    "complete_matrix": True,
                    "experiment_id": scope["experiment_id"],
                    "experiment_config_sha256": scope["_sha256"],
                    "run_id": "fixture-run",
                    "phase": "pilot",
                    "population_id": "pilot_eval_13",
                    "baseline_indicator": scope["baseline_indicator"],
                    "full_context_indicators": scope["full_context_indicators"],
                    "intervention_indicators": scope["intervention_indicators"],
                    "minutes_system_prompt": scope["minutes_system_prompt"],
                    "decoding": scope["decoding"],
                    "minutes_model_sha256": "3" * 64,
                    "minutes_tokenizer_sha256": "4" * 64,
                    "artifacts": {
                        arm: fingerprint_artifact_path(path)
                        for arm, path in generation_paths.items()
                    },
                }
            )
            release_path.write_text(json.dumps(release), encoding="utf-8")
        release = json.loads(release_path.read_text(encoding="utf-8"))
        return {
            "reference": reference_path,
            "reference_manifest": reference_manifest_path,
            "release": release_path,
            "release_payload": release,
            "generation_paths": generation_paths,
        }

    def _arm_fixture(self, root: Path, arm: str) -> tuple[Path, Path]:
        scope = _load_scoped_experiment(SCOPE_FILE)
        dependencies = self._dependency_fixture(root)
        regime, strategy = ARM_SPECS[arm]
        replicate_count = len(scope["decoding"][regime]["replicate_seeds"])
        rows = []
        for indicator_index, indicator in enumerate(scope["intervention_indicators"]):
            for meeting_index, meeting_date in enumerate(
                scope["populations"]["pilot_eval_13"]["meeting_dates"]
            ):
                for section_name in scope["section_names"]:
                    for replicate_id in range(replicate_count):
                        base = 0.001 * (indicator_index + 1) + 0.0001 * meeting_index
                        if regime == "stochastic":
                            base += 0.00001 * replicate_id
                        delta = base if strategy == DELETION_STRATEGY else base * 0.25
                        rows.append(
                            {
                                "target_mode": "actual-minutes",
                                "indicator": indicator,
                                "context": "pilot_eval_13",
                                "meeting_date": meeting_date,
                                "section_name": section_name,
                                "replicate_id": str(replicate_id),
                                "similarity_full": 0.8,
                                "similarity_masked": 0.8 - delta,
                                "delta": delta,
                                "masking_strategy": strategy,
                            }
                        )
        rows_path = root / f"{arm}.rows.jsonl"
        audit_path = root / f"{arm}.audit.json"
        _write_jsonl(rows_path, rows)
        audit = {
            "target_mode": "actual-minutes",
            "excluded_rows": 0,
            "scored_pairs": len(rows),
            "row_output_file": str(rows_path.resolve()),
            "reference_file": str(dependencies["reference"].resolve()),
            "reference_file_sha256": sha256_file(dependencies["reference"]),
            "reference_key_fields": ["meeting_date", "section_name"],
            "reference_text_field": "response",
            "reference_duplicate_policy": "error",
            "embedding_model_path": "/frozen/embedding",
            "embedding_model_artifact": {
                "sha256": "2" * 64,
                "kind": "directory",
            },
            "embedding_max_tokens": 4096,
            "embedding_long_text_policy": "chunk-mean",
            "baseline_indicator": "None",
            "generation_manifest": {
                "masking_strategy": strategy,
                "path": str(dependencies["generation_paths"][arm].resolve()),
                "sha256": sha256_file(dependencies["generation_paths"][arm]),
                "generation_model_artifact": {"sha256": "3" * 64},
                "generation_tokenizer_artifact": {"sha256": "4" * 64},
            },
            "scoped_experiment": {
                "sha256": scope["_sha256"],
                "experiment_id": scope["experiment_id"],
                "population_id": "pilot_eval_13",
                "regime": regime,
                "intervention_strategy": strategy,
                "reference_manifest": {
                    "path": str(dependencies["reference_manifest"].resolve()),
                    "sha256": sha256_file(dependencies["reference_manifest"]),
                },
                "generation_release": {
                    "path": str(dependencies["release"].resolve()),
                    "sha256": sha256_file(dependencies["release"]),
                    "payload_sha256": dependencies["release_payload"]["integrity"][
                        "payload_sha256"
                    ],
                    "run_id": "fixture-run",
                },
            },
        }
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        return rows_path, audit_path

    def test_writes_four_arm_report_with_frozen_aggregation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            arm_rows = {}
            arm_audits = {}
            for arm in ARM_SPECS:
                arm_rows[arm], arm_audits[arm] = self._arm_fixture(root, arm)

            manifest = summarize_canonical_loo(
                scope_manifest_file=SCOPE_FILE,
                population_id="pilot_eval_13",
                arm_rows=arm_rows,
                arm_audits=arm_audits,
                output_dir=root / "report",
            )

            overall_path = Path(manifest["outputs"]["overall_csv"]["path"])
            meeting_path = Path(manifest["outputs"]["meeting_level_csv"]["path"])
            with overall_path.open(newline="", encoding="utf-8") as handle:
                overall = list(csv.DictReader(handle))
            with meeting_path.open(newline="", encoding="utf-8") as handle:
                meeting = list(csv.DictReader(handle))
            self.assertEqual(len(overall), 12)
            self.assertEqual(len(meeting), 156)
            first = overall[0]
            self.assertAlmostEqual(
                float(first["delete_minus_neutral_mean"]),
                float(first["deletion_mean"]) - float(first["neutral_mean"]),
            )
            self.assertEqual(first["holm_family_size"], "6")
            self.assertEqual(manifest["bootstrap"]["draws"], 10_000)
            self.assertEqual(manifest["inference_unit"], "meeting")
            self.assertEqual(manifest["legacy_or_remaining20_pooling"], "forbidden")
            self.assertIn("integrity", manifest)
            report_text = Path(
                manifest["outputs"]["report_markdown"]["path"]
            ).read_text(encoding="utf-8")
            self.assertIn("Pilot — exploratory/protocol validation", report_text)
            self.assertIn("Meeting SD", report_text)
            self.assertIn("Auxiliary neutral and deletion−neutral", report_text)
            self.assertIn("two-sided meeting-level one-sample t-tests", report_text)
            self.assertIn("| Delete−Neutral |", report_text)
            manifest_path = root / "report/pilot_eval_13_loo_six_indicator_manifest.json"
            with patch(
                "jobs.eval.summarize_canonical_loo.validate_scoped_release_binding",
                side_effect=self._mock_release_validation,
            ):
                validation = validate_report_manifest(
                    manifest_path,
                    scope_manifest_file=SCOPE_FILE,
                    population_id="pilot_eval_13",
                )
            self.assertEqual(validation["status"], "validated")
            Path(manifest["outputs"]["report_markdown"]["path"]).write_text(
                report_text + "tampered\n",
                encoding="utf-8",
            )
            with patch(
                "jobs.eval.summarize_canonical_loo.validate_scoped_release_binding",
                side_effect=self._mock_release_validation,
            ):
                with self.assertRaisesRegex(ValueError, "changed or is missing"):
                    validate_report_manifest(
                        manifest_path,
                        scope_manifest_file=SCOPE_FILE,
                        population_id="pilot_eval_13",
                    )

    def test_aggregation_averages_sections_before_replicates(self):
        rows = []
        for replicate_id, values in (("0", [1.0, 2.0, 3.0]), ("1", [3.0, 4.0, 5.0])):
            for section_name, value in zip(["A", "B", "C"], values, strict=True):
                rows.append(
                    {
                        "indicator": "I",
                        "meeting_date": "2024-01-01",
                        "replicate_id": replicate_id,
                        "section_name": section_name,
                        "delta": value,
                    }
                )
        scope = {
            "section_names": ["A", "B", "C"],
            "intervention_indicators": ["I"],
            "populations": {"p": {"meeting_dates": ["2024-01-01"]}},
            "decoding": {"stochastic": {"replicate_seeds": [1, 2]}},
        }
        meeting = aggregate_sections_replicates_meetings(
            rows,
            scope=scope,
            population_id="p",
            regime="stochastic",
        )
        self.assertEqual(len(meeting), 1)
        self.assertAlmostEqual(float(meeting.iloc[0]["delta"]), 3.0)

    def test_joint_bootstrap_reuses_identical_meeting_draws_for_all_series(self):
        base = np.arange(13, dtype=float)
        transformed = 2.0 * base + 3.0
        intervals = _joint_bootstrap_metrics(
            {("A", "deletion"): base, ("B", "neutral"): transformed},
            draws=10_000,
            seed=20260728,
        )
        base_interval = intervals[("A", "deletion")]
        transformed_interval = intervals[("B", "neutral")]
        self.assertAlmostEqual(transformed_interval[0], 2.0 * base_interval[0] + 3.0)
        self.assertAlmostEqual(transformed_interval[1], 2.0 * base_interval[1] + 3.0)

    def test_zero_variance_p_values_keep_the_six_test_holm_family(self):
        self.assertEqual(_t_test(np.zeros(13)), (0.0, 1.0, "zero_variance_zero_mean"))
        t_stat, p_value, status = _t_test(np.ones(13))
        self.assertIsNone(t_stat)
        self.assertEqual(p_value, 0.0)
        self.assertEqual(status, "zero_variance_nonzero_constant_infinite_t_limit")
        adjusted = _holm_six_deletion_family([1.0, 0.5, 0.2, 0.1, 0.0, 0.8])
        self.assertEqual(len(adjusted), 6)
        with self.assertRaisesRegex(ValueError, "six finite"):
            _holm_six_deletion_family([1.0, 0.5, None, 0.1, 0.0, 0.8])

    def test_rejects_mixed_reference_or_model_provenance_across_arms(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            arm_rows = {}
            arm_audits = {}
            for arm in ARM_SPECS:
                arm_rows[arm], arm_audits[arm] = self._arm_fixture(root, arm)
            mixed_audit = json.loads(
                arm_audits["neutral_primary"].read_text(encoding="utf-8")
            )
            mixed_audit["reference_file_sha256"] = "9" * 64
            arm_audits["neutral_primary"].write_text(
                json.dumps(mixed_audit), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "incompatible target/model/release"):
                summarize_canonical_loo(
                    scope_manifest_file=SCOPE_FILE,
                    population_id="pilot_eval_13",
                    arm_rows=arm_rows,
                    arm_audits=arm_audits,
                    output_dir=root / "report",
                )

    def test_resume_rejects_tampered_reference_or_scoped_release_dependency(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            arm_rows = {}
            arm_audits = {}
            for arm in ARM_SPECS:
                arm_rows[arm], arm_audits[arm] = self._arm_fixture(root, arm)
            summarize_canonical_loo(
                scope_manifest_file=SCOPE_FILE,
                population_id="pilot_eval_13",
                arm_rows=arm_rows,
                arm_audits=arm_audits,
                output_dir=root / "report",
            )
            report_manifest = (
                root / "report/pilot_eval_13_loo_six_indicator_manifest.json"
            )
            dependencies = self._dependency_fixture(root)
            reference_path = dependencies["reference"]
            original_reference = reference_path.read_bytes()
            reference_path.write_bytes(original_reference + b"\n")
            with patch(
                "jobs.eval.summarize_canonical_loo.validate_scoped_release_binding",
                side_effect=self._mock_release_validation,
            ):
                with self.assertRaises(ValueError):
                    validate_report_manifest(
                        report_manifest,
                        scope_manifest_file=SCOPE_FILE,
                        population_id="pilot_eval_13",
                    )
            reference_path.write_bytes(original_reference)

            release_path = dependencies["release"]
            release_path.write_bytes(release_path.read_bytes() + b"\n")
            with patch(
                "jobs.eval.summarize_canonical_loo.validate_scoped_release_binding",
                side_effect=self._mock_release_validation,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Scoped generation release dependency changed",
                ):
                    validate_report_manifest(
                        report_manifest,
                        scope_manifest_file=SCOPE_FILE,
                        population_id="pilot_eval_13",
                    )


if __name__ == "__main__":
    unittest.main()
