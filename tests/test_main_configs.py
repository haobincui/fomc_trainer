from pathlib import Path
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestMainConfigs(unittest.TestCase):
    def test_prompt_pipeline_defaults_to_gemini_thought_channel(self):
        config = yaml.safe_load((REPO_ROOT / "configs/main/prompt_pipeline.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["response_template"], "gemma4_thought_channel")

    def test_active_training_configs_use_native_gemini_thinking_prompt(self):
        config_paths = [
            REPO_ROOT / "configs/main/analysis_sft.yaml",
            REPO_ROOT / "configs/main/analysis_grpo.yaml",
            REPO_ROOT / "configs/main/decision_sft.yaml",
            REPO_ROOT / "configs/main/decision_grpo.yaml",
            REPO_ROOT / "configs/main/minutes_alignment_sft.yaml",
        ]

        for path in config_paths:
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                system_prompt = config["system_prompt"].strip()
                self.assertTrue(system_prompt.startswith("<|think|>"))
                self.assertNotIn("<think>", system_prompt)
                self.assertNotIn("<answer>", system_prompt)

    def test_training_configs_point_to_processed_train_roots(self):
        config_paths = [
            REPO_ROOT / "configs/main/analysis_sft.yaml",
            REPO_ROOT / "configs/main/analysis_grpo.yaml",
            REPO_ROOT / "configs/main/decision_sft.yaml",
            REPO_ROOT / "configs/main/decision_grpo.yaml",
            REPO_ROOT / "configs/main/minutes_alignment_sft.yaml",
        ]

        for path in config_paths:
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                dataset_name = config["dataset_name"]
                self.assertTrue(dataset_name.startswith("dataset/processed/train/"))
                self.assertNotIn("/main/", dataset_name)

    def test_generate_entrypoint_uses_native_gemini_prompt(self):
        source = (REPO_ROOT / "src/open_r1/generate.py").read_text(encoding="utf-8")
        self.assertIn('_SYSTEM_PROMPT = """<|think|>', source)
        self.assertNotIn("<think>", source)
        self.assertNotIn("<answer>", source)


if __name__ == "__main__":
    unittest.main()
