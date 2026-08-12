import subprocess
import tempfile
import unittest
from pathlib import Path

from jobs.generation.mask_generation import (
    build_parser,
    select_intervention_prompt_files,
    validate_generation_spec_binding,
)
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import (
    build_generation_spec,
    write_frozen_generation_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestInterventionPromptSelection(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.roster = {
            "baseline_indicator": "None",
            "indicators": ["Indicator-A", "Indicator-B", "Indicator-C"],
            "contexts": ["pilot_eval_13"],
        }
        self.baseline_path = self._write_prompt(
            "None",
            "full baseline with Indicator-A, Indicator-B, and Indicator-C",
        )
        self.input_prompt_files = [
            str(self.baseline_path),
            str(self._write_prompt("Indicator-A", "without Indicator-A")),
            str(self._write_prompt("Indicator-B", "without Indicator-B")),
            str(self._write_prompt("Indicator-C", "without Indicator-C")),
        ]

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_prompt(self, indicator: str, prompt: str) -> Path:
        path = self.root / f"{indicator}_masked_pilot_eval_13.jsonl"
        path.write_text(prompt + "\n", encoding="utf-8")
        return path

    def test_partial_selection_preserves_full_baseline(self):
        baseline_before = self.baseline_path.read_bytes()

        selected, scope = select_intervention_prompt_files(
            input_prompt_files=self.input_prompt_files,
            roster=self.roster,
            include_intervention_indicators=["Indicator-C", "Indicator-A"],
        )

        self.assertEqual(
            [Path(path).name for path in selected],
            [
                "None_masked_pilot_eval_13.jsonl",
                "Indicator-A_masked_pilot_eval_13.jsonl",
                "Indicator-C_masked_pilot_eval_13.jsonl",
            ],
        )
        self.assertEqual(self.baseline_path.read_bytes(), baseline_before)
        self.assertEqual(scope["mode"], "partial_intervention_shard")
        self.assertEqual(
            scope["full_roster_indicators"],
            ["Indicator-A", "Indicator-B", "Indicator-C"],
        )
        self.assertEqual(
            scope["generated_intervention_indicators"],
            ["Indicator-A", "Indicator-C"],
        )
        self.assertEqual(
            scope["omitted_intervention_indicators"],
            ["Indicator-B"],
        )
        self.assertTrue(scope["baseline_contains_full_roster"])
        self.assertFalse(scope["standalone_canonical_release_allowed"])

    def test_rejects_unknown_duplicate_and_baseline_selectors(self):
        cases = [
            (["Unknown"], "outside the frozen roster"),
            (["Indicator-A", "Indicator-A"], "contains duplicates"),
            (["None"], "must not be passed"),
        ]
        for selection, message in cases:
            with self.subTest(selection=selection):
                with self.assertRaisesRegex(ValueError, message):
                    select_intervention_prompt_files(
                        input_prompt_files=self.input_prompt_files,
                        roster=self.roster,
                        include_intervention_indicators=selection,
                    )

    def test_cli_selector_is_repeatable(self):
        args = build_parser().parse_args(
            [
                "--input-folder",
                "prompts",
                "--model",
                "model",
                "--output-dir",
                "outputs",
                "--roster-file",
                "roster.json",
                "--include-intervention-indicator",
                "Indicator-C",
                "--include-intervention-indicator",
                "Indicator-A",
            ]
        )

        self.assertEqual(
            args.include_intervention_indicators,
            ["Indicator-C", "Indicator-A"],
        )

    def test_generation_spec_requires_named_minutes_artifacts(self):
        minutes_model = self.root / "minutes-model"
        analysis_model = self.root / "analysis-model"
        minutes_tokenizer = self.root / "minutes-tokenizer"
        analysis_tokenizer = self.root / "analysis-tokenizer"
        for path, content in (
            (minutes_model, b"minutes-model"),
            (analysis_model, b"analysis-model"),
            (minutes_tokenizer, b"minutes-tokenizer"),
            (analysis_tokenizer, b"analysis-tokenizer"),
        ):
            path.mkdir()
            (path / "artifact.bin").write_bytes(content)
        prompt_folder = self.root / "spec-prompts"
        prompt_folder.mkdir()
        (prompt_folder / "None_masked_pilot_eval_13.jsonl").write_text(
            "{}\n",
            encoding="utf-8",
        )
        prompt_manifest = self.root / "prompt_manifest.json"
        prompt_manifest.write_text("{}\n", encoding="utf-8")
        spec = build_generation_spec(
            run_id="named-binding-test",
            phase="pilot",
            population_id="pilot_eval_13",
            models={
                "analysis_model": analysis_model,
                "minutes_model": minutes_model,
            },
            tokenizers={
                "analysis_tokenizer": analysis_tokenizer,
                "minutes_tokenizer": minutes_tokenizer,
            },
            sources={
                "prompt_manifest": prompt_manifest,
                "prompt_folder": prompt_folder,
            },
            generation_config={"generation_only": True},
            replicate_seeds=[20260728],
        )
        spec_path = self.root / "generation_spec.json"
        spec_sha256 = write_frozen_generation_spec(spec_path, spec)
        metadata = {
            "sha256": sha256_file(prompt_manifest),
            "population_id": "pilot_eval_13",
        }

        validate_generation_spec_binding(
            generation_spec_file=spec_path,
            expected_file_sha256=spec_sha256,
            model_artifact_sha256=fingerprint_artifact_path(minutes_model)[
                "sha256"
            ],
            tokenizer_artifact_sha256=fingerprint_artifact_path(
                minutes_tokenizer
            )["sha256"],
            prompt_manifest_metadata=metadata,
            input_dir=prompt_folder,
        )
        with self.assertRaisesRegex(
            ValueError,
            "models.minutes_model",
        ):
            validate_generation_spec_binding(
                generation_spec_file=spec_path,
                expected_file_sha256=spec_sha256,
                model_artifact_sha256=fingerprint_artifact_path(
                    analysis_model
                )["sha256"],
                tokenizer_artifact_sha256=fingerprint_artifact_path(
                    analysis_tokenizer
                )["sha256"],
                prompt_manifest_metadata=metadata,
                input_dir=prompt_folder,
            )
        with self.assertRaisesRegex(
            ValueError,
            "tokenizers.minutes_tokenizer",
        ):
            validate_generation_spec_binding(
                generation_spec_file=spec_path,
                expected_file_sha256=spec_sha256,
                model_artifact_sha256=fingerprint_artifact_path(
                    minutes_model
                )["sha256"],
                tokenizer_artifact_sha256=fingerprint_artifact_path(
                    analysis_tokenizer
                )["sha256"],
                prompt_manifest_metadata=metadata,
                input_dir=prompt_folder,
            )


class TestPartialShardLauncherWiring(unittest.TestCase):
    def test_shell_launchers_are_syntactically_valid_and_wire_partial_scope(self):
        generation_script = REPO_ROOT / "run/_run_canonical_loo_generation.sh"
        workflow_script = REPO_ROOT / "run/generate_loo_end_to_end.sh"

        for script in (generation_script, workflow_script):
            with self.subTest(script=script.name):
                subprocess.run(
                    ["bash", "-n", str(script)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

        generation_source = generation_script.read_text(encoding="utf-8")
        workflow_source = workflow_script.read_text(encoding="utf-8")
        self.assertIn("LOO_INTERVENTION_INDICATORS", generation_source)
        self.assertIn("--include-intervention-indicator", generation_source)
        self.assertIn(
            "jobs.generation.finalize_loo_intervention_shard",
            generation_source,
        )
        self.assertIn("standalone_canonical_release=false", generation_source)
        self.assertIn("LOO_INTERVENTION_INDICATORS", workflow_source)
        self.assertIn(
            "Partial intervention shards currently require mode=pilot.",
            workflow_source,
        )
        self.assertIn("intervention_shard_manifest.json", workflow_source)
        self.assertIn("standalone_canonical_release=false", workflow_source)


if __name__ == "__main__":
    unittest.main()
