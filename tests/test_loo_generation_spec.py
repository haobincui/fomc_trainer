import copy
import json
import tempfile
import unittest
from pathlib import Path

from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    SPEC_SCHEMA_VERSION,
    FrozenArtifactMismatchError,
    GenerationSafetyError,
    ManifestIntegrityError,
    build_generation_spec,
    derive_row_seed,
    load_and_validate_generation_spec,
    seal_manifest,
    validate_context_budget,
    validate_generation_completion,
    validate_generation_spec,
    validate_manifest_integrity,
    write_frozen_generation_spec,
)


class TestLooGenerationSpec(unittest.TestCase):
    def _artifacts(self, root: Path) -> tuple[Path, Path, Path]:
        model = root / "model"
        tokenizer = root / "tokenizer"
        model.mkdir()
        tokenizer.mkdir()
        (model / "config.json").write_text('{"model":"frozen"}\n', encoding="utf-8")
        (model / "weights.bin").write_bytes(b"immutable model weights")
        (tokenizer / "tokenizer.json").write_text(
            '{"tokenizer":"frozen"}\n',
            encoding="utf-8",
        )
        source = root / "source.jsonl"
        source.write_text('{"sample_id":"m1::section"}\n', encoding="utf-8")
        return model, tokenizer, source

    def _build_spec(self, root: Path) -> dict:
        model, tokenizer, source = self._artifacts(root)
        return build_generation_spec(
            run_id="pilot-20260728",
            phase="seven-indicator-pilot",
            population_id="evaluation-13",
            models={"minutes_model": model},
            tokenizers={"minutes_tokenizer": tokenizer},
            sources={"canonical_prompts": source},
            generation_config={
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 128,
                "context_limit": 512,
            },
            replicate_seeds=[20260728],
        )

    def test_builds_and_revalidates_frozen_artifact_spec(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = self._build_spec(root)

            payload_digest = validate_generation_spec(spec)

            self.assertEqual(spec["schema_version"], SPEC_SCHEMA_VERSION)
            self.assertEqual(
                payload_digest,
                spec["integrity"]["payload_sha256"],
            )
            self.assertEqual(
                set(spec["frozen_artifacts"]),
                {"schema_version", "models", "tokenizers", "sources"},
            )
            self.assertEqual(
                spec["seed_policy"]["derivation_inputs"],
                ["replicate_seed", "sample_id"],
            )

    def test_changed_frozen_artifact_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = self._build_spec(root)
            model_path = Path(
                spec["frozen_artifacts"]["models"]["minutes_model"]["path"]
            )
            (model_path / "weights.bin").write_bytes(b"changed model weights")

            with self.assertRaisesRegex(
                FrozenArtifactMismatchError,
                "no longer matches",
            ):
                validate_generation_spec(spec)

    def test_manifest_payload_edit_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            spec = self._build_spec(Path(temporary_directory))
            tampered = copy.deepcopy(spec)
            tampered["phase"] = "formal"

            with self.assertRaisesRegex(
                ManifestIntegrityError,
                "modified after it was sealed",
            ):
                validate_generation_spec(tampered, verify_artifact_paths=False)

    def test_external_digest_detects_resealed_manifest(self):
        original = seal_manifest({"schema_version": "example", "value": 1})
        original_digest = original["integrity"]["payload_sha256"]
        resealed = seal_manifest({"schema_version": "example", "value": 2})

        with self.assertRaisesRegex(
            ManifestIntegrityError,
            "externally frozen",
        ):
            validate_manifest_integrity(
                resealed,
                expected_payload_sha256=original_digest,
            )

    def test_frozen_write_and_load_validate_external_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = self._build_spec(root)
            path = root / "run" / "generation_spec.json"

            written_sha256 = write_frozen_generation_spec(path, spec)
            loaded = load_and_validate_generation_spec(
                path,
                expected_file_sha256=written_sha256,
            )

            self.assertEqual(loaded, spec)
            self.assertEqual(sha256_file(path), written_sha256)

            resealed = seal_manifest(
                {
                    key: value
                    for key, value in spec.items()
                    if key != "integrity"
                }
                | {"phase": "formal"}
            )
            path.write_text(json.dumps(resealed), encoding="utf-8")
            with self.assertRaisesRegex(
                ManifestIntegrityError,
                "externally frozen file digest",
            ):
                load_and_validate_generation_spec(
                    path,
                    verify_artifact_paths=False,
                    expected_file_sha256=written_sha256,
                )

    def test_refuses_to_overwrite_a_different_frozen_spec(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = self._build_spec(root)
            path = root / "generation_spec.json"
            write_frozen_generation_spec(path, spec)
            path.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ManifestIntegrityError,
                "Refusing to overwrite",
            ):
                write_frozen_generation_spec(path, spec)


class TestRowSeedDerivation(unittest.TestCase):
    def test_seed_is_stable_and_independent_of_order_and_batching(self):
        replicate_seed = 20260728
        sample_ids = ["m1::views", "m2::economic", "m3::financial"]
        forward = {
            sample_id: derive_row_seed(replicate_seed, sample_id)
            for sample_id in sample_ids
        }
        reversed_order = {
            sample_id: derive_row_seed(replicate_seed, sample_id)
            for sample_id in reversed(sample_ids)
        }

        self.assertEqual(forward, reversed_order)
        self.assertEqual(
            forward["m1::views"],
            derive_row_seed(replicate_seed, "m1::views"),
        )
        self.assertEqual(len(set(forward.values())), len(sample_ids))
        self.assertTrue(all(0 <= seed < 2**31 for seed in forward.values()))

    def test_seed_changes_with_replicate_or_sample(self):
        baseline = derive_row_seed(20260728, "m1::views")

        self.assertNotEqual(baseline, derive_row_seed(21260728, "m1::views"))
        self.assertNotEqual(baseline, derive_row_seed(20260728, "m2::views"))


class TestGenerationSafetyValidation(unittest.TestCase):
    def test_context_boundary_and_successful_stop_are_valid(self):
        context_audit = validate_context_budget(
            input_token_count=384,
            max_new_tokens=128,
            context_limit=512,
            consumed_input_token_count=384,
        )
        completion_audit = validate_generation_completion(
            input_token_count=384,
            output_token_count=64,
            max_new_tokens=128,
            context_limit=512,
            finish_reason="stop",
            consumed_input_token_count=384,
        )

        self.assertEqual(context_audit["unused_context_tokens"], 0)
        self.assertEqual(completion_audit["finish_reason"], "stop")
        self.assertFalse(completion_audit["consumed_full_output_budget"])

    def test_context_overflow_is_rejected(self):
        with self.assertRaisesRegex(
            GenerationSafetyError,
            "exceeds the model context window",
        ):
            validate_context_budget(
                input_token_count=385,
                max_new_tokens=128,
                context_limit=512,
            )

    def test_input_truncation_and_token_count_mismatch_are_rejected(self):
        with self.assertRaisesRegex(GenerationSafetyError, "truncation is forbidden"):
            validate_context_budget(
                input_token_count=100,
                max_new_tokens=50,
                context_limit=200,
                input_was_truncated=True,
            )
        with self.assertRaisesRegex(
            GenerationSafetyError,
            "Consumed input token count differs",
        ):
            validate_context_budget(
                input_token_count=100,
                max_new_tokens=50,
                context_limit=200,
                consumed_input_token_count=99,
            )

    def test_token_limit_and_unknown_finish_reasons_are_rejected(self):
        with self.assertRaisesRegex(GenerationSafetyError, "token limit"):
            validate_generation_completion(
                input_token_count=100,
                output_token_count=50,
                max_new_tokens=50,
                context_limit=200,
                finish_reason="length",
            )
        with self.assertRaisesRegex(GenerationSafetyError, "Unknown"):
            validate_generation_completion(
                input_token_count=100,
                output_token_count=25,
                max_new_tokens=50,
                context_limit=200,
                finish_reason="cancelled",
            )


if __name__ == "__main__":
    unittest.main()
