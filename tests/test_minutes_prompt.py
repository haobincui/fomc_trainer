import json
import tempfile
import unittest
from pathlib import Path

from open_r1.minutes_prompt import (
    CANONICAL_MINUTES_PROMPT_SPEC,
    MINUTES_SYSTEM_PROMPT_SHA256,
    load_minutes_prompt_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestMinutesPromptContract(unittest.TestCase):
    def test_base_and_scoped_configs_bind_the_exact_same_prompt(self):
        base_spec, _, _ = load_minutes_prompt_config(
            REPO_ROOT / "configs/main/canonical_loo_generation.json"
        )
        scoped_spec, scoped_binding, scoped_payload = load_minutes_prompt_config(
            REPO_ROOT / "configs/main/loo_experiment_legacy6.json"
        )

        self.assertEqual(base_spec, CANONICAL_MINUTES_PROMPT_SPEC)
        self.assertEqual(scoped_spec, CANONICAL_MINUTES_PROMPT_SPEC)
        self.assertEqual(scoped_spec.sha256, MINUTES_SYSTEM_PROMPT_SHA256)
        self.assertEqual(scoped_binding["schema_version"], "loo-scoped-experiment-v1")
        self.assertEqual(scoped_binding["experiment_id"], "legacy6-full26-v1")
        self.assertEqual(len(scoped_payload["full_context_indicators"]), 26)
        self.assertEqual(len(scoped_payload["intervention_indicators"]), 6)
        self.assertNotIn("yield_curve", scoped_payload["intervention_indicators"])

    def test_text_or_declared_hash_tampering_is_rejected(self):
        source = json.loads(
            (
                REPO_ROOT / "configs/main/loo_experiment_legacy6.json"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tampered.json"
            source["minutes_system_prompt"]["text"] += " Changed."
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256 does not match"):
                load_minutes_prompt_config(path)


if __name__ == "__main__":
    unittest.main()
