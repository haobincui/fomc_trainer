import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generate_new_response import (
    ROW_SEED_POLICY_SAMPLE,
    derive_row_seed,
    generate_new_response,
)
from jobs.eval.eval_leave_one_out import run_paired_evaluation
from jobs.generation.mask_generation import (
    build_intervention_manifest,
    run_mask_generation,
    validate_output_file,
)


class ConstantTripletScorer:
    def score_triplets(
        self,
        targets,
        full_outputs,
        masked_outputs,
        *,
        target_mode,
    ):
        return [(1.0, 0.8, 0.8) for _ in targets]


class TestGenerateNewResponse(unittest.TestCase):
    @patch("generate_new_response.generate_responses")
    def test_failed_batch_outputs_are_not_counted_as_success(self, mock_generate_responses):
        mock_generate_responses.return_value = ["good output", "Failed", ""]
        input_rows = [
            {"prompt": "p1", "response": "r1"},
            {"prompt": "p2", "response": "r2"},
            {"prompt": "p3", "response": "r3"},
        ]

        results = generate_new_response(input_rows, "models/test-model", batch_size=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["generated"], "good output")
        self.assertNotIn("generated", input_rows[0])
        self.assertNotIn("generated", input_rows[1])

    @patch("generate_new_response.generate_responses")
    def test_legacy_argument_order_is_normalized(self, mock_generate_responses):
        mock_generate_responses.return_value = ["generated text"]
        input_rows = [{"prompt": "prompt", "response": "target"}]

        results = generate_new_response(input_rows, "output/test.jsonl", "models/test-model", batch_size=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["generated"], "generated text")

    @patch("generate_new_response.generate_responses")
    def test_generation_metadata_records_replicate_and_actual_batch_seed(
        self,
        mock_generate_responses,
    ):
        mock_generate_responses.return_value = ["generated text"]
        input_rows = [{"prompt": "prompt", "response": "target"}]

        results = generate_new_response(
            input_rows,
            "models/test-model",
            batch_size=1,
            seed=100,
            replicate_id=2,
            temperature=0.0,
            top_p=1.0,
            generation_metadata={
                "indicator": "GDP-Growth",
                "masking_strategy": "indicator_block_deletion",
            },
        )

        self.assertEqual(results[0]["replicate_id"], "2")
        self.assertEqual(results[0]["generation_seed"], 100)
        self.assertEqual(results[0]["generation_position"], 0)
        self.assertEqual(results[0]["generation_batch_size"], 1)
        self.assertEqual(results[0]["generation_model"], "models/test-model")
        self.assertEqual(results[0]["indicator"], "GDP-Growth")
        self.assertEqual(results[0]["decoding_temperature"], 0.0)
        self.assertEqual(results[0]["decoding_top_p"], 1.0)
        self.assertEqual(mock_generate_responses.call_args.kwargs["seed"], 100)

    def test_resume_validation_rejects_truncated_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "generated.jsonl"
            output_file.write_text(
                json.dumps(
                    {
                        "generation_position": 0,
                        "replicate_id": "0",
                        "generation_seed": 100,
                        "generation_model": "models/test-model",
                        "generation_batch_size": 1,
                        "decoding_temperature": 0.0,
                        "decoding_top_p": 1.0,
                        "max_new_tokens": 128,
                        "masking_strategy": "indicator_block_deletion",
                        "indicator": "GDP-Growth",
                        "evaluation_context": "test",
                        "source_file_sha256": "abc",
                        "source_row_count": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            valid = validate_output_file(
                str(output_file),
                replicate_id=0,
                replicate_seed=100,
                expected_row_count=2,
                source_file_sha256="abc",
                batch_size=1,
                model_path="models/test-model",
                model_artifact_sha256="a" * 64,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=128,
                masking_strategy="indicator_block_deletion",
                indicator="GDP-Growth",
                evaluation_context="test",
                expected_prompt_rows=[
                    {
                        "sample_id": "s1",
                        "meeting_date": "2024-01-31",
                        "section_name": "Section A",
                        "prompt": "prompt one",
                    },
                    {
                        "sample_id": "s2",
                        "meeting_date": "2024-03-20",
                        "section_name": "Section A",
                        "prompt": "prompt two",
                    },
                ],
            )

            self.assertFalse(valid)

    def test_source_index_is_preserved_separately_from_generation_position(self):
        with patch(
            "generate_new_response.generate_responses",
            return_value=["generated text"],
        ):
            results = generate_new_response(
                [{"index": 77, "prompt": "prompt", "response": "target"}],
                "models/test-model",
                batch_size=1,
                seed=100,
                replicate_id=0,
            )

        self.assertEqual(results[0]["index"], 77)
        self.assertEqual(results[0]["source_index"], 77)
        self.assertEqual(results[0]["generation_position"], 0)

    @patch("generate_new_response.generate_responses")
    def test_sample_id_seed_policy_is_batch_and_indicator_independent(
        self,
        mock_generate_responses,
    ):
        def respond(prompts, model_path, **kwargs):
            return [
                {
                    "text": f"generated:{prompt}",
                    "finish_reason": "stop",
                    "prompt_token_count": 10,
                    "prompt_preflight_token_count": 10,
                    "output_token_count": 2,
                    "input_was_truncated": False,
                }
                for prompt in prompts
            ]

        mock_generate_responses.side_effect = respond
        rows = [
            {
                "sample_id": "2024-01-31::economic",
                "prompt": "first",
            },
            {
                "sample_id": "2024-03-20::economic",
                "prompt": "second",
            },
        ]

        first = generate_new_response(
            rows,
            "models/test-model",
            batch_size=1,
            seed=20260728,
            seed_policy=ROW_SEED_POLICY_SAMPLE,
            tokenizer_path="models/test-tokenizer",
        )
        second = generate_new_response(
            list(reversed(rows)),
            "models/test-model",
            batch_size=2,
            seed=20260728,
            seed_policy=ROW_SEED_POLICY_SAMPLE,
            tokenizer_path="models/test-tokenizer",
        )

        first_seeds = {
            row["sample_id"]: row["generation_seed"] for row in first
        }
        second_seeds = {
            row["sample_id"]: row["generation_seed"] for row in second
        }
        self.assertEqual(first_seeds, second_seeds)
        self.assertEqual(
            first_seeds["2024-01-31::economic"],
            derive_row_seed(20260728, "2024-01-31::economic"),
        )
        self.assertTrue(
            all(
                call.kwargs["seed"] is None
                for call in mock_generate_responses.call_args_list
            )
        )
        self.assertTrue(
            all(
                call.kwargs["row_seeds"]
                for call in mock_generate_responses.call_args_list
            )
        )
        self.assertEqual(
            first[0]["generation_tokenizer"],
            "models/test-tokenizer",
        )
        self.assertEqual(first[0]["generation_finish_reason"], "stop")
        self.assertEqual(first[0]["prompt_token_count"], 10)
        self.assertEqual(first[0]["prompt_preflight_token_count"], 10)
        self.assertEqual(first[0]["output_token_count"], 2)
        self.assertFalse(first[0]["input_was_truncated"])
        self.assertEqual(
            first[0]["generated_sha256"],
            hashlib.sha256(first[0]["generated"].encode()).hexdigest(),
        )

    def test_intervention_manifest_rejects_non_deletion_prompt_change(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prompts = root / "prompts"
            prompts.mkdir()
            roster_file = root / "roster.json"
            roster_file.write_text(
                json.dumps(
                    {
                        "schema_version": "loo-intervention-roster-v1",
                        "baseline_indicator": "None",
                        "contexts": ["test"],
                        "indicators": ["GDP-Growth"],
                        "indicator_markers": {
                            "GDP-Growth": ["GDP"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            base = {
                "sample_id": "s1",
                "meeting_date": "2024-01-31",
                "section_name": "Section A",
                "prompt": "prefix GDP suffix",
            }
            (prompts / "None_masked_test.jsonl").write_text(
                json.dumps(base) + "\n",
                encoding="utf-8",
            )
            (prompts / "GDP-Growth_masked_test.jsonl").write_text(
                json.dumps(dict(base, prompt="prefix CHANGED suffix")) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Invalid masking intervention"):
                build_intervention_manifest(
                    input_dir=prompts.resolve(),
                    input_prompt_files=sorted(path.resolve() for path in prompts.glob("*.jsonl")),
                    roster_file=roster_file.resolve(),
                )

    def test_intervention_manifest_rejects_missing_indicator_family(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prompts = root / "prompts"
            prompts.mkdir()
            roster_file = root / "roster.json"
            roster_file.write_text(
                json.dumps(
                    {
                        "schema_version": "loo-intervention-roster-v1",
                        "baseline_indicator": "None",
                        "contexts": ["test"],
                        "indicators": ["GDP-Growth"],
                        "indicator_markers": {
                            "GDP-Growth": ["GDP"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (prompts / "None_masked_test.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "s1",
                        "meeting_date": "2024-01-31",
                        "section_name": "Section A",
                        "prompt": "full prompt",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "frozen intervention roster"):
                build_intervention_manifest(
                    input_dir=prompts.resolve(),
                    input_prompt_files=sorted(path.resolve() for path in prompts.glob("*.jsonl")),
                    roster_file=roster_file.resolve(),
                )

    def test_intervention_manifest_rejects_wrong_indicator_block(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prompts = root / "prompts"
            prompts.mkdir()
            roster_file = root / "roster.json"
            roster_file.write_text(
                json.dumps(
                    {
                        "schema_version": "loo-intervention-roster-v1",
                        "baseline_indicator": "None",
                        "contexts": ["test"],
                        "indicators": ["GDP-Growth"],
                        "indicator_markers": {
                            "GDP-Growth": ["GDP"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            base = {
                "sample_id": "s1",
                "meeting_date": "2024-01-31",
                "section_name": "Section A",
                "prompt": "prefix\nUnrelated block",
            }
            (prompts / "None_masked_test.jsonl").write_text(
                json.dumps(base) + "\n",
                encoding="utf-8",
            )
            (prompts / "GDP-Growth_masked_test.jsonl").write_text(
                json.dumps(dict(base, prompt="prefix")) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "indicator marker"):
                build_intervention_manifest(
                    input_dir=prompts.resolve(),
                    input_prompt_files=sorted(
                        path.resolve() for path in prompts.glob("*.jsonl")
                    ),
                    roster_file=roster_file.resolve(),
                )

    def test_intervention_manifest_rejects_duplicate_masked_prompts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prompts = root / "prompts"
            prompts.mkdir()
            roster_file = root / "roster.json"
            roster_file.write_text(
                json.dumps(
                    {
                        "schema_version": "loo-intervention-roster-v1",
                        "baseline_indicator": "None",
                        "contexts": ["test"],
                        "indicators": ["GDP-Growth", "Consumer-Price-Index-(CPI)"],
                        "indicator_markers": {
                            "GDP-Growth": ["GDP"],
                            "Consumer-Price-Index-(CPI)": ["CPI"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            base = {
                "sample_id": "s1",
                "meeting_date": "2024-01-31",
                "section_name": "Section A",
                "prompt": "prefix\nGDP and CPI shared block",
            }
            (prompts / "None_masked_test.jsonl").write_text(
                json.dumps(base) + "\n",
                encoding="utf-8",
            )
            for indicator in ("GDP-Growth", "Consumer-Price-Index-(CPI)"):
                (prompts / f"{indicator}_masked_test.jsonl").write_text(
                    json.dumps(dict(base, prompt="prefix")) + "\n",
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(ValueError, "Duplicate masked prompt"):
                build_intervention_manifest(
                    input_dir=prompts.resolve(),
                    input_prompt_files=sorted(
                        path.resolve() for path in prompts.glob("*.jsonl")
                    ),
                    roster_file=roster_file.resolve(),
                )

    def test_intervention_manifest_rejects_overlapping_indicator_blocks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prompts = root / "prompts"
            prompts.mkdir()
            roster_file = root / "roster.json"
            roster_file.write_text(
                json.dumps(
                    {
                        "schema_version": "loo-intervention-roster-v1",
                        "baseline_indicator": "None",
                        "contexts": ["test"],
                        "indicators": ["GDP-Growth", "Unemployment-Rate"],
                        "indicator_markers": {
                            "GDP-Growth": ["GDP"],
                            "Unemployment-Rate": ["Unemployment"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            base = {
                "sample_id": "s1",
                "meeting_date": "2024-01-31",
                "section_name": "Section A",
                "prompt": (
                    "prefix\nGDP block\nUnemployment block\nsuffix"
                ),
            }
            (prompts / "None_masked_test.jsonl").write_text(
                json.dumps(base) + "\n",
                encoding="utf-8",
            )
            (prompts / "GDP-Growth_masked_test.jsonl").write_text(
                json.dumps(dict(base, prompt="prefix\nsuffix")) + "\n",
                encoding="utf-8",
            )
            (
                prompts / "Unemployment-Rate_masked_test.jsonl"
            ).write_text(
                json.dumps(
                    dict(base, prompt="prefix\nGDP block\nsuffix")
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "Overlapping indicator-block deletions",
            ):
                build_intervention_manifest(
                    input_dir=prompts.resolve(),
                    input_prompt_files=sorted(
                        path.resolve() for path in prompts.glob("*.jsonl")
                    ),
                    roster_file=roster_file.resolve(),
                )

    @patch("generate_new_response.generate_responses")
    def test_mask_generation_writes_complete_hashed_manifest(
        self,
        mock_generate_responses,
    ):
        mock_generate_responses.return_value = ["generated one", "generated two"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prompts = root / "prompts"
            outputs = root / "outputs"
            model = root / "model"
            roster_file = root / "roster.json"
            prompts.mkdir()
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            roster_file.write_text(
                json.dumps(
                    {
                        "schema_version": "loo-intervention-roster-v1",
                        "roster_id": "test-roster",
                        "baseline_indicator": "None",
                        "contexts": ["test"],
                        "indicators": ["GDP-Growth"],
                        "indicator_markers": {
                            "GDP-Growth": ["GDP"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            prompt_rows = [
                {
                    "sample_id": "s1",
                    "index": 10,
                    "meeting_date": "2024-01-31",
                    "section_name": "Section A",
                    "prompt": "prompt one\nGDP block one",
                    "response": "target one",
                },
                {
                    "sample_id": "s2",
                    "index": 11,
                    "meeting_date": "2024-03-20",
                    "section_name": "Section A",
                    "prompt": "prompt two\nGDP block two",
                    "response": "target two",
                },
            ]
            (prompts / "None_masked_test.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in prompt_rows),
                encoding="utf-8",
            )
            masked_prompt_rows = [
                dict(prompt_rows[0], prompt="prompt one"),
                dict(prompt_rows[1], prompt="prompt two"),
            ]
            (prompts / "GDP-Growth_masked_test.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in masked_prompt_rows),
                encoding="utf-8",
            )

            run_mask_generation(
                input_folder=str(prompts),
                model_path=str(model),
                simulation_step=1,
                output_dir=str(outputs),
                roster_file=str(roster_file),
                batch_size=2,
                seed=100,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=128,
            )

            manifest = json.loads(
                (outputs / "generation_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], "loo-generation-v4")
            self.assertEqual(len(manifest["input_artifacts"]), 2)
            self.assertEqual(len(manifest["expected_artifacts"]), 2)
            self.assertEqual(len(manifest["model_artifact"]["sha256"]), 64)
            intervention_manifest = json.loads(
                (outputs / "intervention_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                intervention_manifest["validation"],
                "exact_single_contiguous_prompt_deletion",
            )
            self.assertEqual(len(intervention_manifest["interventions"]), 2)
            self.assertTrue(
                intervention_manifest["distinct_masked_prompt_per_indicator"]
            )
            for artifact in manifest["expected_artifacts"]:
                self.assertEqual(artifact["source_row_count"], 2)
                self.assertEqual(len(artifact["output_sha256"]), 64)

            audit = run_paired_evaluation(
                input_folder=outputs,
                summary_output_file=root / "internal_summary.jsonl",
                target_mode="full-output",
                embedding_model_path="fake-embedding",
                embedding_model_sha256="b" * 64,
                scorer=ConstantTripletScorer(),
                bootstrap_samples=20,
            )
            self.assertEqual(audit["scored_pairs"], 2)
            self.assertEqual(audit["generation_manifest"]["status"], "validated")

            full_artifact = outputs / "None_masked_test_0.jsonl"
            tampered_rows = [
                json.loads(line)
                for line in full_artifact.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            tampered_rows[0]["prompt"] = "tampered prompt"
            tampered_rows[0]["source_prompt_sha256"] = hashlib.sha256(
                b"tampered prompt"
            ).hexdigest()
            full_artifact.write_text(
                "".join(json.dumps(row) + "\n" for row in tampered_rows),
                encoding="utf-8",
            )
            manifest = json.loads(
                (outputs / "generation_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            for artifact in manifest["expected_artifacts"]:
                if artifact["relative_path"] == full_artifact.name:
                    artifact["output_sha256"] = hashlib.sha256(
                        full_artifact.read_bytes()
                    ).hexdigest()
            (outputs / "generation_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "attested intervention prompt"):
                run_paired_evaluation(
                    input_folder=outputs,
                    summary_output_file=root / "tampered_summary.jsonl",
                    target_mode="full-output",
                    embedding_model_path="fake-embedding",
                    embedding_model_sha256="b" * 64,
                    scorer=ConstantTripletScorer(),
                    bootstrap_samples=20,
                )


if __name__ == "__main__":
    unittest.main()
