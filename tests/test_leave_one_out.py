import json
import math
import tempfile
import unittest
from pathlib import Path

from jobs.eval.eval_leave_one_out import (
    _holm_adjust,
    discover_generated_artifacts,
    load_and_validate_generation_manifest,
    pair_generated_artifacts,
    run_paired_evaluation,
    summarise_scored_rows,
)
from open_r1.validator.leave_one_out import (
    leave_one_out_metrics_from_similarities,
    score_leave_one_out,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class MappingTripletScorer:
    def __init__(self, scores: dict[tuple[str, str], float]):
        self.scores = scores

    def score_triplets(self, targets, full_outputs, masked_outputs, *, target_mode):
        results = []
        for target, full, masked in zip(
            targets,
            full_outputs,
            masked_outputs,
            strict=True,
        ):
            similarity_full = 1.0 if target_mode == "full-output" else self.scores[(target, full)]
            similarity_masked = self.scores[(target, masked)]
            self_similarity = self.scores[(full, masked)]
            results.append((similarity_full, similarity_masked, self_similarity))
        return results


class TestLeaveOneOutMetric(unittest.TestCase):
    def test_signed_delta_and_distance_identity(self):
        metrics = leave_one_out_metrics_from_similarities(
            0.8,
            0.6,
            self_similarity=0.7,
        )

        self.assertAlmostEqual(metrics["delta"], 0.2)
        self.assertAlmostEqual(metrics["distance_full"], 0.2)
        self.assertAlmostEqual(metrics["distance_masked"], 0.4)
        self.assertAlmostEqual(
            metrics["distance_masked"],
            metrics["distance_full"] + metrics["delta"],
        )
        self.assertAlmostEqual(metrics["self_distance"], 0.3)
        self.assertAlmostEqual(metrics["self_similarity"], 0.7)

    def test_negative_delta_is_preserved(self):
        metrics = leave_one_out_metrics_from_similarities(0.6, 0.75)
        self.assertAlmostEqual(metrics["delta"], -0.15)

    def test_single_row_scorer_uses_fixed_external_target(self):
        score_map = {
            ("actual", "full"): 0.8,
            ("actual", "masked"): 0.6,
            ("full", "masked"): 0.5,
        }
        metrics = score_leave_one_out(
            full_output="full",
            masked_output="masked",
            target="actual",
            scorer=lambda target, generated: score_map[(target, generated)],
        )

        self.assertAlmostEqual(metrics["delta"], 0.2)
        self.assertNotAlmostEqual(metrics["delta"], 1.0 - 0.6)


class TestPairedLeaveOneOutEvaluation(unittest.TestCase):
    def test_artifact_discovery_accepts_complete_row_metadata_without_legacy_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            _write_jsonl(
                folder / "arbitrary.jsonl",
                [
                    {
                        "indicator": "None",
                        "evaluation_context": "test",
                        "replicate_id": "0",
                        "sample_id": "s1",
                        "meeting_date": "2024-01-31",
                        "section_name": "Section A",
                        "generated": "full",
                    }
                ],
            )

            artifact = discover_generated_artifacts(folder)[0]

            self.assertEqual(artifact.indicator, "None")
            self.assertEqual(artifact.context, "test")
            self.assertEqual(artifact.replicate_id, "0")

    def test_artifact_discovery_fails_on_any_invalid_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            _write_jsonl(
                folder / "None_masked_test_0.jsonl",
                [{"index": 0, "generated": "full"}],
            )
            (folder / "GDP-Growth_masked_test_0.jsonl").write_text(
                "{not json}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "artifacts"):
                discover_generated_artifacts(folder)

    def test_artifact_discovery_keeps_all_replicates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            base_row = {
                "meeting_date": "2024-01-31",
                "section_name": "Section A",
                "index": 0,
                "generated": "full",
            }
            masked_row = dict(base_row, generated="masked")
            for replicate in (0, 1):
                _write_jsonl(
                    folder / f"None_masked_after_2009_{replicate}.jsonl",
                    [dict(base_row, replicate_id=str(replicate))],
                )
                _write_jsonl(
                    folder / f"GDP-Growth_masked_after_2009_{replicate}.jsonl",
                    [dict(masked_row, replicate_id=str(replicate))],
                )

            artifacts = discover_generated_artifacts(folder)
            pairs = pair_generated_artifacts(artifacts)

            self.assertEqual(len(artifacts), 4)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(
                {masked.replicate_id for _, masked in pairs},
                {"0", "1"},
            )

    def test_pairing_rejects_incomplete_replicate_coverage(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            row = {
                "meeting_date": "2024-01-31",
                "section_name": "Section A",
                "index": 0,
                "generated": "text",
            }
            for replicate in (0, 1):
                _write_jsonl(
                    folder / f"None_masked_test_{replicate}.jsonl",
                    [dict(row, replicate_id=str(replicate))],
                )
            _write_jsonl(
                folder / "GDP-Growth_masked_test_0.jsonl",
                [dict(row, replicate_id="0")],
            )

            artifacts = discover_generated_artifacts(folder)
            with self.assertRaisesRegex(ValueError, "complete baseline replicate"):
                pair_generated_artifacts(artifacts)

    def test_canonical_run_requires_generation_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            row = {
                "sample_id": "s1",
                "meeting_date": "2024-01-31",
                "section_name": "Section A",
                "replicate_id": "0",
                "generated": "text",
            }
            _write_jsonl(folder / "None_masked_test_0.jsonl", [row])
            _write_jsonl(folder / "GDP-Growth_masked_test_0.jsonl", [row])

            with self.assertRaisesRegex(FileNotFoundError, "Generation manifest"):
                run_paired_evaluation(
                    input_folder=folder,
                    summary_output_file=folder.parent / "summary.jsonl",
                    target_mode="full-output",
                    embedding_model_path="fake",
                    embedding_model_sha256="b" * 64,
                    scorer=MappingTripletScorer({("text", "text"): 1.0}),
                )

    def test_generation_manifest_validates_complete_inventory_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            row = {
                "sample_id": "s1",
                "meeting_date": "2024-01-31",
                "section_name": "Section A",
                "replicate_id": "0",
                "generated": "text",
            }
            for name in (
                "None_masked_test_0.jsonl",
                "GDP-Growth_masked_test_0.jsonl",
            ):
                _write_jsonl(folder / name, [row])
            artifacts = discover_generated_artifacts(folder)
            manifest = {
                "schema_version": "loo-generation-v2",
                "expected_artifacts": [
                    {
                        "relative_path": artifact.path.name,
                        "indicator": artifact.indicator,
                        "context": artifact.context,
                        "replicate_id": artifact.replicate_id,
                        "source_row_count": artifact.row_count,
                        "output_sha256": artifact.sha256,
                    }
                    for artifact in artifacts
                ],
            }
            (folder / "generation_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            loaded, audit = load_and_validate_generation_manifest(
                folder,
                artifacts,
                required=False,
            )

            self.assertEqual(loaded["schema_version"], "loo-generation-v2")
            self.assertEqual(audit["status"], "validated")
            self.assertEqual(
                audit["intervention_manifest"]["status"],
                "not_available_in_legacy_generation_manifest",
            )

    def test_pairing_rejects_generation_configuration_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            common = {
                "sample_id": "s1",
                "meeting_date": "2024-01-31",
                "section_name": "Section A",
                "replicate_id": "0",
                "generation_seed": 42,
                "generation_batch_size": 1,
                "decoding_temperature": 0.0,
                "decoding_top_p": 1.0,
                "max_new_tokens": 128,
                "masking_strategy": "indicator_block_deletion",
            }
            _write_jsonl(
                folder / "None_masked_test_0.jsonl",
                [dict(common, generation_model="model-a", generated="full")],
            )
            _write_jsonl(
                folder / "GDP-Growth_masked_test_0.jsonl",
                [dict(common, generation_model="model-b", generated="masked")],
            )

            with self.assertRaisesRegex(ValueError, "generation_configuration_mismatch"):
                run_paired_evaluation(
                    input_folder=folder,
                    summary_output_file=folder.parent / "summary.jsonl",
                    target_mode="full-output",
                    embedding_model_path="fake",
                    embedding_model_sha256="b" * 64,
                    scorer=MappingTripletScorer({("full", "masked"): 0.5}),
                    require_generation_manifest=False,
                )

    def test_pairing_rejects_conflicting_sample_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            _write_jsonl(
                folder / "None_masked_test_0.jsonl",
                [
                    {
                        "sample_id": "s1",
                        "meeting_date": "2024-01-31",
                        "section_name": "Section A",
                        "replicate_id": "0",
                        "generated": "full",
                    }
                ],
            )
            _write_jsonl(
                folder / "GDP-Growth_masked_test_0.jsonl",
                [
                    {
                        "sample_id": "s1",
                        "meeting_date": "2024-03-20",
                        "section_name": "Section A",
                        "replicate_id": "0",
                        "generated": "masked",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "row_identity_metadata_mismatch"):
                run_paired_evaluation(
                    input_folder=folder,
                    summary_output_file=folder.parent / "summary.jsonl",
                    target_mode="full-output",
                    embedding_model_path="fake",
                    embedding_model_sha256="b" * 64,
                    scorer=MappingTripletScorer({("full", "masked"): 0.5}),
                    require_generation_manifest=False,
                )

    def test_rows_without_sample_or_source_index_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            row = {
                "meeting_date": "2024-01-31",
                "section_name": "Section A",
                "replicate_id": "0",
                "generated": "text",
            }
            _write_jsonl(folder / "None_masked_test_0.jsonl", [row])
            _write_jsonl(folder / "GDP-Growth_masked_test_0.jsonl", [row])

            with self.assertRaisesRegex(ValueError, "no stable key"):
                run_paired_evaluation(
                    input_folder=folder,
                    summary_output_file=folder.parent / "summary.jsonl",
                    target_mode="full-output",
                    embedding_model_path="fake",
                    embedding_model_sha256="b" * 64,
                    scorer=MappingTripletScorer({("text", "text"): 1.0}),
                    require_generation_manifest=False,
                )

    def test_rows_with_blank_section_are_rejected_even_with_sample_id(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            row = {
                "sample_id": "s1",
                "meeting_date": "2024-01-31",
                "section_name": " ",
                "replicate_id": "0",
                "generated": "text",
            }
            _write_jsonl(folder / "None_masked_test_0.jsonl", [row])
            _write_jsonl(folder / "GDP-Growth_masked_test_0.jsonl", [row])

            with self.assertRaisesRegex(
                ValueError,
                "non-empty meeting_date and section_name",
            ):
                run_paired_evaluation(
                    input_folder=folder,
                    summary_output_file=folder.parent / "summary.jsonl",
                    target_mode="full-output",
                    embedding_model_path="fake",
                    embedding_model_sha256="b" * 64,
                    scorer=MappingTripletScorer({("text", "text"): 1.0}),
                    require_generation_manifest=False,
                )

    def test_pairing_rejects_generation_position_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            common = {
                "sample_id": "s1",
                "meeting_date": "2024-01-31",
                "section_name": "Section A",
                "replicate_id": "0",
                "generation_seed": 42,
                "generation_model": "model-a",
                "generation_batch_size": 2,
                "decoding_temperature": 0.0,
                "decoding_top_p": 1.0,
                "max_new_tokens": 128,
                "masking_strategy": "indicator_block_deletion",
            }
            _write_jsonl(
                folder / "None_masked_test_0.jsonl",
                [dict(common, generation_position=0, generated="full")],
            )
            _write_jsonl(
                folder / "GDP-Growth_masked_test_0.jsonl",
                [dict(common, generation_position=1, generated="masked")],
            )

            with self.assertRaisesRegex(
                ValueError,
                "generation_configuration_mismatch",
            ):
                run_paired_evaluation(
                    input_folder=folder,
                    summary_output_file=folder.parent / "summary.jsonl",
                    target_mode="full-output",
                    embedding_model_path="fake",
                    embedding_model_sha256="b" * 64,
                    scorer=MappingTripletScorer({("full", "masked"): 0.5}),
                    require_generation_manifest=False,
                )

    def test_actual_minutes_path_scores_full_minus_masked_by_key(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            generated = root / "generated"
            full_rows = [
                {
                    "meeting_date": "2024-01-31",
                    "section_name": "Section A",
                    "index": 0,
                    "replicate_id": "0",
                    "generation_seed": 42,
                    "generated": "<|channel>thought\nhidden full\n<channel|>full answer one",
                },
                {
                    "meeting_date": "2024-03-20",
                    "section_name": "Section A",
                    "index": 1,
                    "replicate_id": "0",
                    "generation_seed": 42,
                    "generated": "<answer>full answer two</answer>",
                },
            ]
            # Reverse physical order to prove that the join is keyed, not positional.
            masked_rows = [
                {
                    "meeting_date": "2024-03-20",
                    "section_name": "Section A",
                    "index": 1,
                    "replicate_id": "0",
                    "generation_seed": 42,
                    "generated": "<think>hidden</think><answer>masked answer two</answer>",
                },
                {
                    "meeting_date": "2024-01-31",
                    "section_name": "Section A",
                    "index": 0,
                    "replicate_id": "0",
                    "generation_seed": 42,
                    "generated": "<answer>masked answer one</answer>",
                },
            ]
            _write_jsonl(generated / "None_masked_after_2009_0.jsonl", full_rows)
            _write_jsonl(generated / "GDP-Growth_masked_after_2009_0.jsonl", masked_rows)

            reference_file = root / "actual_minutes.jsonl"
            _write_jsonl(
                reference_file,
                [
                    {
                        "meeting_date": "2024-01-31",
                        "section_name": "Section A",
                        "response": "actual one",
                    },
                    {
                        "meeting_date": "2024-03-20",
                        "section_name": "Section A",
                        "response": "actual two",
                    },
                ],
            )
            scorer = MappingTripletScorer(
                {
                    ("actual one", "full answer one"): 0.8,
                    ("actual one", "masked answer one"): 0.6,
                    ("full answer one", "masked answer one"): 0.5,
                    ("actual two", "full answer two"): 0.7,
                    ("actual two", "masked answer two"): 0.75,
                    ("full answer two", "masked answer two"): 0.65,
                }
            )
            summary_file = root / "actual_summary.jsonl"
            audit = run_paired_evaluation(
                input_folder=generated,
                summary_output_file=summary_file,
                target_mode="actual-minutes",
                embedding_model_path="fake-embedding-model",
                embedding_model_sha256="b" * 64,
                reference_file=reference_file,
                scorer=scorer,
                bootstrap_samples=100,
                bootstrap_seed=9,
                require_generation_manifest=False,
            )

            row_file = root / "actual_summary.rows.jsonl"
            rows = _read_jsonl(row_file)
            rows_by_date = {row["meeting_date"]: row for row in rows}
            self.assertAlmostEqual(rows_by_date["2024-01-31"]["delta"], 0.2)
            self.assertAlmostEqual(rows_by_date["2024-03-20"]["delta"], -0.05)
            self.assertEqual(rows_by_date["2024-01-31"]["full_output"], "full answer one")
            self.assertEqual(
                rows_by_date["2024-01-31"]["embedding_model_sha256"],
                "b" * 64,
            )
            self.assertEqual(
                audit["embedding_model_artifact"]["sha256"],
                "b" * 64,
            )
            self.assertEqual(audit["scored_pairs"], 2)
            self.assertEqual(audit["excluded_rows"], 0)

    def test_summary_averages_replicates_before_meetings(self):
        rows = []
        for meeting_date, replicate_id, delta in [
            ("2024-01-31", "0", 0.2),
            ("2024-01-31", "1", 0.4),
            ("2024-03-20", "0", 0.0),
        ]:
            rows.append(
                {
                    "target_mode": "actual-minutes",
                    "context": "after_2009",
                    "indicator": "GDP-Growth",
                    "section_name": "Section A",
                    "meeting_date": meeting_date,
                    "replicate_id": replicate_id,
                    "row_key": ["sample_id", f"{meeting_date}-{replicate_id}"],
                    "similarity_full": 0.8,
                    "similarity_masked": 0.8 - delta,
                    "delta": delta,
                }
            )

        summary = summarise_scored_rows(
            rows,
            bootstrap_samples=100,
            bootstrap_seed=3,
        )[0]

        self.assertEqual(summary["n_pairs"], 3)
        self.assertEqual(summary["n_meetings"], 2)
        self.assertEqual(summary["n_replicates"], 2)
        self.assertAlmostEqual(summary["mean_delta"], 0.15)

    def test_summary_keeps_evaluation_contexts_separate(self):
        rows = []
        for context, delta in (("train", 0.1), ("test", 0.9)):
            rows.append(
                {
                    "target_mode": "actual-minutes",
                    "context": context,
                    "indicator": "GDP-Growth",
                    "section_name": "Section A",
                    "meeting_date": "2024-01-31",
                    "replicate_id": "0",
                    "row_key": ["sample_id", f"{context}-s1"],
                    "similarity_full": 0.9,
                    "similarity_masked": 0.9 - delta,
                    "delta": delta,
                }
            )

        summaries = summarise_scored_rows(
            rows,
            bootstrap_samples=20,
            bootstrap_seed=3,
        )

        self.assertEqual(len(summaries), 2)
        self.assertEqual(
            {row["context"]: row["mean_delta"] for row in summaries},
            {"test": 0.9, "train": 0.1},
        )

    def test_summary_equal_weights_replicates_within_meeting(self):
        rows = []
        for meeting, replicate, sample, delta in (
            ("2024-01-31", "0", "a", 0.1),
            ("2024-01-31", "0", "b", 0.3),
            ("2024-01-31", "1", "a", 0.8),
            ("2024-03-20", "0", "a", 0.0),
        ):
            rows.append(
                {
                    "target_mode": "actual-minutes",
                    "context": "test",
                    "indicator": "GDP-Growth",
                    "section_name": "Section A",
                    "meeting_date": meeting,
                    "replicate_id": replicate,
                    "row_key": ["sample_id", f"{meeting}-{sample}"],
                    "similarity_full": 0.9,
                    "similarity_masked": 0.9 - delta,
                    "delta": delta,
                }
            )

        summary = summarise_scored_rows(
            rows,
            bootstrap_samples=20,
            bootstrap_seed=3,
        )[0]

        # Meeting 1: mean(mean(0.1, 0.3), 0.8) = 0.5; meeting 2: 0.
        self.assertAlmostEqual(summary["mean_delta"], 0.25)
        self.assertEqual(summary["replicates_per_meeting_min"], 1)
        self.assertEqual(summary["replicates_per_meeting_max"], 2)

    def test_single_meeting_has_no_bootstrap_interval(self):
        summary = summarise_scored_rows(
            [
                {
                    "target_mode": "actual-minutes",
                    "context": "test",
                    "indicator": "GDP-Growth",
                    "section_name": "Section A",
                    "meeting_date": "2024-01-31",
                    "replicate_id": "0",
                    "row_key": ["sample_id", "s1"],
                    "similarity_full": 0.9,
                    "similarity_masked": 0.8,
                    "delta": 0.1,
                }
            ],
            bootstrap_samples=20,
            bootstrap_seed=3,
        )[0]

        self.assertTrue(math.isnan(summary["ci_lower"]))
        self.assertTrue(math.isnan(summary["ci_upper"]))
        self.assertEqual(summary["inference_status"], "insufficient_meeting_clusters")

    def test_holm_adjustment_matches_known_vector(self):
        adjusted = _holm_adjust([0.01, 0.04, 0.03, math.nan])

        self.assertEqual(adjusted[:3], [0.03, 0.06, 0.06])
        self.assertTrue(math.isnan(adjusted[3]))


if __name__ == "__main__":
    unittest.main()
