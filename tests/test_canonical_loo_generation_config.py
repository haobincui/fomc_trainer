import json
import subprocess
import unittest
from pathlib import Path

from jobs.main.run_pipeline import build_parser


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestCanonicalLooGenerationConfig(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (REPO_ROOT / "configs/main/canonical_loo_generation.json").read_text(
                encoding="utf-8"
            )
        )

    def test_populations_are_disjoint_ordered_and_match_run_rosters(self):
        populations = self.config["populations"]
        pilot_dates = populations["pilot_eval_13"]["meeting_dates"]
        formal_dates = populations["formal_test_13"]["meeting_dates"]

        self.assertEqual(len(pilot_dates), 13)
        self.assertEqual(len(formal_dates), 13)
        self.assertEqual(pilot_dates, sorted(pilot_dates))
        self.assertEqual(formal_dates, sorted(formal_dates))
        self.assertFalse(set(pilot_dates) & set(formal_dates))

        for population_id, filename in (
            ("pilot_eval_13", "loo_population_pilot_eval_13.json"),
            ("formal_test_13", "loo_population_formal_test_13.json"),
        ):
            roster = json.loads(
                (REPO_ROOT / "configs/main" / filename).read_text(encoding="utf-8")
            )
            self.assertEqual(roster["population_id"], population_id)
            self.assertEqual(
                roster["meeting_dates"],
                populations[population_id]["meeting_dates"],
            )

    def test_rosters_and_decoding_match_generation_only_design(self):
        indicator_roster = json.loads(
            (REPO_ROOT / "configs/main/leave_one_out_roster.json").read_text(
                encoding="utf-8"
            )
        )
        section_roster = json.loads(
            (REPO_ROOT / "configs/main/loo_sections.json").read_text(encoding="utf-8")
        )

        self.assertTrue(self.config["generation_only"])
        self.assertFalse(self.config["training_performed"])
        self.assertIn(
            "output/checkpoints/recovered/",
            self.config["artifacts"]["minutes_model"],
        )
        self.assertNotEqual(
            self.config["artifacts"]["minutes_model"],
            "../fomc_trainer_back/fomc_trainer/output/merged/"
            "llama_sft_synthetic_20250526",
        )
        provenance = self.config["checkpoint_provenance"]
        self.assertEqual(
            provenance["minutes_artifact_id"],
            "eval-minutes-sft-from-chk1",
        )
        self.assertEqual(
            provenance["runtime_verification"],
            "required-before-generation",
        )
        self.assertEqual(
            provenance["minutes_model_sha256"],
            "ac176afc59f73eccbded8e02b4eec7dd1de8b252c0b20352f5074cb9e02077ea",
        )
        self.assertEqual(
            provenance["minutes_tokenizer_sha256"],
            provenance["minutes_model_sha256"],
        )
        for key in (
            "minutes_manifest_sha256",
            "minutes_manifest_payload_sha256",
            "minutes_model_sha256",
            "minutes_tokenizer_sha256",
        ):
            self.assertRegex(provenance[key], r"^[0-9a-f]{64}$")
        self.assertEqual(len(indicator_roster["indicators"]), 26)
        self.assertEqual(len(section_roster["sections"]), 3)
        self.assertEqual(
            self.config["interventions"],
            [
                "indicator_block_deletion",
                "indicator_block_neutral_replacement",
            ],
        )
        self.assertEqual(
            self.config["decoding"]["minutes_primary"]["replicate_seeds"],
            [20260728],
        )
        self.assertEqual(
            self.config["decoding"]["indicator_analysis"]["max_token_limit_errors"],
            2,
        )
        self.assertEqual(
            self.config["decoding"]["indicator_analysis"]["max_new_tokens"],
            8192,
        )
        self.assertEqual(
            self.config["decoding"]["indicator_analysis"][
                "requested_max_output_tokens"
            ],
            4096,
        )
        self.assertEqual(
            self.config["invariants"]["indicator_analysis_token_limit_finish"],
            {
                "policy": "bounded-token-limit-errors-v1",
                "max_errors": 2,
            },
        )
        self.assertEqual(
            self.config["invariants"]["minutes_token_limit_finish"],
            "forbidden",
        )
        self.assertEqual(
            self.config["decoding"]["minutes_stochastic_robustness"]["replicate_seeds"],
            [
                20260729,
                21260729,
                22260729,
                23260729,
                24260729,
            ],
        )
        self.assertEqual(
            {
                key: self.config["decoding"]["minutes_primary"][key]
                for key in ("max_new_tokens", "max_model_len")
            },
            {"max_new_tokens": 8192, "max_model_len": 16384},
        )
        self.assertEqual(
            {
                key: self.config["decoding"]["minutes_stochastic_robustness"][key]
                for key in ("max_new_tokens", "max_model_len")
            },
            {"max_new_tokens": 8192, "max_model_len": 16384},
        )
        self.assertEqual(
            self.config["execution"],
            {
                "gpu_policy": "single-physical-gpu-by-nvidia-smi-index-v1",
                "physical_gpu_index": 1,
                "expected_visible_cuda_devices": 1,
                "tensor_parallel_size": 1,
            },
        )
        projection = self.config["analysis_projection"]
        self.assertEqual(projection["policy"], "deepseek-final-answer-after-think-v1")
        self.assertEqual(projection["source_field"], "generated")
        self.assertEqual(projection["output_field"], "minutes_analysis")
        self.assertEqual(projection["delimiter"], "</think>")
        self.assertEqual(projection["required_delimiter_count"], 1)
        self.assertFalse(projection["allow_plain_text_fallback"])
        self.assertEqual(projection["truncation"], "forbidden")
        self.assertEqual(projection["secondary_generation"], "forbidden")

    def test_background_launchers_are_valid_and_contain_no_training_stage(self):
        scripts = [
            REPO_ROOT / "run/_run_canonical_loo_generation.sh",
            REPO_ROOT / "run/generate_loo_pilot.sh",
            REPO_ROOT / "run/generate_loo_formal.sh",
            REPO_ROOT / "run/generate_loo_pilot_remaining20_recovered.sh",
            REPO_ROOT / "run/generate_loo_end_to_end.sh",
        ]
        subprocess.run(
            ["bash", "-n", *(str(path) for path in scripts)],
            check=True,
            cwd=REPO_ROOT,
        )
        generation_sources = "\n".join(
            path.read_text(encoding="utf-8") for path in scripts
        )
        for forbidden in (
            "jobs.train",
            "accelerate launch",
            "jobs.merge_model",
            "api_key",
            "import requests",
        ):
            self.assertNotIn(forbidden, generation_sources)
        end_to_end_source = scripts[-1].read_text(encoding="utf-8")
        for frozen_input in (
            "FROZEN_REGISTRY",
            "FROZEN_INDICATOR_ROSTER",
            "FROZEN_SECTION_ROSTER",
            "FROZEN_GENERATION_CONFIG",
            "FROZEN_PILOT_POPULATION",
            "FROZEN_FORMAL_POPULATION",
            "FROZEN_PILOT_RELEASE",
        ):
            self.assertIn(frozen_input, end_to_end_source)
        self.assertIn("flock -n", generation_sources)
        self.assertIn("export PYTHONPATH=", generation_sources)
        self.assertIn("torch.cuda.is_available()", end_to_end_source)
        self.assertIn("import vllm", end_to_end_source)
        self.assertIn(
            'analysis["max_token_limit_errors"]',
            generation_sources,
        )
        self.assertIn("--max-token-limit-errors", generation_sources)
        self.assertIn("jobs.generation.project_indicator_analysis", generation_sources)
        self.assertIn("--analysis-field", generation_sources)
        self.assertIn("PROJECTION_OUTPUT_FIELD", generation_sources)
        self.assertIn("--projection-manifest", generation_sources)
        self.assertIn(
            "jobs.main.checkpoint_provenance verify-artifact",
            generation_sources,
        )
        self.assertIn(
            "minutes_checkpoint_runtime_verification=",
            generation_sources,
        )

    def test_main_entrypoint_exposes_generation_only_stages(self):
        parser = build_parser()
        analysis = parser.parse_args(
            [
                "generate-loo-analysis",
                "--input",
                "input.jsonl",
                "--population",
                "population.json",
                "--ledger-manifest",
                "ledger.json",
                "--snapshot-manifest",
                "snapshot.json",
                "--source-registry",
                "registry.json",
                "--model",
                "analysis-model",
                "--tokenizer",
                "analysis-tokenizer",
                "--output-dir",
                "analysis-output",
            ]
        )
        prompts = parser.parse_args(
            [
                "build-loo-prompts",
                "--analysis-blocks",
                "analysis.jsonl",
                "--population",
                "population.json",
                "--tokenizer",
                "minutes-tokenizer",
                "--output-dir",
                "prompts",
            ]
        )
        projection = parser.parse_args(
            [
                "project-loo-analysis",
                "--input",
                "analysis.jsonl",
                "--analysis-manifest",
                "analysis_manifest.json",
                "--tokenizer",
                "minutes-tokenizer",
                "--output-dir",
                "projection",
            ]
        )

        self.assertEqual(analysis.command, "generate-loo-analysis")
        self.assertEqual(analysis.max_new_tokens, 8192)
        self.assertEqual(analysis.requested_max_output_tokens, 4096)
        self.assertEqual(analysis.max_token_limit_errors, 0)
        self.assertEqual(prompts.command, "build-loo-prompts")
        self.assertEqual(prompts.analysis_field, "minutes_analysis")
        self.assertEqual(prompts.minutes_max_new_tokens, 8192)
        self.assertEqual(prompts.minutes_max_model_len, 16384)
        self.assertEqual(projection.command, "project-loo-analysis")
        self.assertEqual(projection.output_field, "minutes_analysis")


if __name__ == "__main__":
    unittest.main()
