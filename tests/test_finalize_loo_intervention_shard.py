import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jobs.generation.finalize_loo_intervention_shard import (
    EXPECTED_RUNS,
    build_intervention_shard_manifest,
    validate_generation_completion_manifest,
)
from jobs.generation.mask_generation import (
    apply_completion_policy_to_output,
)
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class TestFinalizeLooInterventionShard(unittest.TestCase):
    def test_validates_and_recomputes_token_limit_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "Indicator-A_0.jsonl"
            generated = "truncated output"
            row = {
                "sample_id": "s1",
                "meeting_date": "2024-01-31",
                "section_name": "Section A",
                "generation_position": 0,
                "generation_finish_reason": "length",
                "prompt_token_count": 100,
                "prompt_preflight_token_count": 100,
                "output_token_count": 50,
                "input_was_truncated": False,
                "generated": generated,
                "generated_sha256": hashlib.sha256(
                    generated.encode("utf-8")
                ).hexdigest(),
            }
            output.write_text(json.dumps(row) + "\n", encoding="utf-8")
            summary = apply_completion_policy_to_output(
                output,
                max_new_tokens=50,
                max_model_len=200,
                token_limit_policy="exclude",
            )
            exclusion = {
                **summary["exclusions"][0],
                "artifact_relative_path": output.name,
                "indicator": "Indicator-A",
                "context": "test",
                "replicate_id": "0",
            }
            artifact = {
                "path": str(output),
                "relative_path": output.name,
                "indicator": "Indicator-A",
                "context": "test",
                "replicate_id": "0",
                "attempted_row_count": 1,
                "scorable_row_count": 0,
                "excluded_row_count": 1,
                "completion_exclusions": [exclusion],
                "output_sha256": sha256_file(output),
            }
            manifest = {
                "status": "complete_with_exclusions",
                "generation_attempts_complete": True,
                "scorable_population_complete": False,
                "max_new_tokens": 50,
                "max_model_len": 200,
                "completion_validation": {
                    "policy": "record-and-exclude-token-limit-v1",
                    "token_limit_policy": "exclude",
                    "attempted_row_count": 1,
                    "passed_row_count": 0,
                    "excluded_row_count": 1,
                    "exclusions": [exclusion],
                },
            }

            observed = validate_generation_completion_manifest(
                manifest=manifest,
                expected_artifacts=[artifact],
                label="test-run",
            )
            self.assertEqual(observed["excluded_row_count"], 1)

            manifest["completion_validation"]["excluded_row_count"] = 0
            with self.assertRaisesRegex(ValueError, "summary mismatch"):
                validate_generation_completion_manifest(
                    manifest=manifest,
                    expected_artifacts=[artifact],
                    label="test-run",
                )

    def test_seals_partial_shard_without_claiming_full_release(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = root / "generation_spec.json"
            prompt = root / "prompt_manifest.json"
            analysis = root / "analysis_manifest.json"
            projection = root / "projection_manifest.json"
            minutes_model = {
                "sha256": "1" * 64,
                "kind": "directory",
                "file_count": 2,
                "total_bytes": 100,
                "algorithm": "fixture-directory-fingerprint",
            }
            minutes_tokenizer = {
                "sha256": "2" * 64,
                "kind": "directory",
                "file_count": 3,
                "total_bytes": 200,
                "algorithm": "fixture-directory-fingerprint",
            }
            _write_json(
                spec,
                {
                    "run_id": "pilot-shard",
                    "phase": "pilot",
                    "population_id": "pilot_eval_13",
                    "frozen_artifacts": {
                        "models": {
                            "analysis_model": {"sha256": "3" * 64},
                            "minutes_model": minutes_model,
                        },
                        "tokenizers": {
                            "analysis_tokenizer": {"sha256": "4" * 64},
                            "minutes_tokenizer": minutes_tokenizer,
                        },
                    },
                },
            )
            _write_json(
                prompt,
                {
                    "baseline_indicator": "None",
                    "indicators": ["Indicator-A", "Indicator-B"],
                },
            )
            _write_json(analysis, {"status": "complete"})
            _write_json(projection, {"status": "complete"})

            execution_environment = {
                "physical_gpu_index": "1",
                "physical_gpu_uuid": "GPU-test",
                "cuda_visible_devices": "1",
                "declared_visible_device_count": "1",
                "tensor_parallel_size": 1,
            }
            scope = {
                "schema_version": "loo-intervention-generation-scope-v1",
                "mode": "partial_intervention_shard",
                "baseline_indicator": "None",
                "full_roster_indicators": ["Indicator-A", "Indicator-B"],
                "generated_intervention_indicators": ["Indicator-A"],
                "omitted_intervention_indicators": ["Indicator-B"],
                "baseline_contains_full_roster": True,
                "standalone_canonical_release_allowed": False,
            }
            for run_name, expectation in EXPECTED_RUNS.items():
                output_dir = root / "generations" / run_name
                artifacts = []
                for indicator in ("None", "Indicator-A"):
                    for replicate in range(expectation["replicate_count"]):
                        artifact = output_dir / f"{indicator}_{replicate}.jsonl"
                        artifact.parent.mkdir(parents=True, exist_ok=True)
                        artifact.write_text(
                            json.dumps({"generated": indicator}) + "\n",
                            encoding="utf-8",
                        )
                        artifacts.append(
                            {
                                "path": str(artifact),
                                "indicator": indicator,
                                "output_sha256": sha256_file(artifact),
                            }
                        )
                _write_json(
                    output_dir / "generation_manifest.json",
                    {
                        "schema_version": "loo-generation-v4",
                        "masking_strategy": expectation["strategy"],
                        "simulation_step": expectation["replicate_count"],
                        "temperature": expectation["temperature"],
                        "top_p": expectation["top_p"],
                        "intervention_scope": scope,
                        "generation_spec": {
                            "file_sha256": sha256_file(spec),
                        },
                        "prompt_manifest": {
                            "sha256": sha256_file(prompt),
                        },
                        "model_artifact": minutes_model,
                        "model_artifact_after_generation": minutes_model,
                        "tokenizer_artifact": minutes_tokenizer,
                        "tokenizer_artifact_after_generation": minutes_tokenizer,
                        "input_artifacts": [
                            {"indicator": "None"},
                            {"indicator": "Indicator-A"},
                        ],
                        "expected_artifacts": artifacts,
                        "execution_environment": execution_environment,
                    },
                )

            manifest = build_intervention_shard_manifest(
                run_root=root,
                generation_spec_file=spec,
                prompt_manifest_file=prompt,
                analysis_manifest_file=analysis,
                projection_manifest_file=projection,
                selected_indicators=["Indicator-A"],
            )

            self.assertEqual(manifest["status"], "partial_complete")
            self.assertFalse(manifest["standalone_canonical_release"])
            self.assertTrue(manifest["baseline_contains_full_roster"])
            self.assertEqual(
                manifest["generated_intervention_indicators"],
                ["Indicator-A"],
            )
            self.assertEqual(
                manifest["omitted_intervention_indicators"],
                ["Indicator-B"],
            )
            validate_manifest_integrity(manifest)

            wrong_manifest_path = (
                root
                / "generations"
                / "deletion_primary"
                / "generation_manifest.json"
            )
            wrong_manifest = json.loads(
                wrong_manifest_path.read_text(encoding="utf-8")
            )
            wrong_model = {
                **minutes_model,
                "sha256": "3" * 64,
            }
            wrong_manifest["model_artifact"] = wrong_model
            wrong_manifest["model_artifact_after_generation"] = wrong_model
            _write_json(wrong_manifest_path, wrong_manifest)
            with self.assertRaisesRegex(
                ValueError,
                "named frozen Minutes artifact",
            ):
                build_intervention_shard_manifest(
                    run_root=root,
                    generation_spec_file=spec,
                    prompt_manifest_file=prompt,
                    analysis_manifest_file=analysis,
                    projection_manifest_file=projection,
                    selected_indicators=["Indicator-A"],
                )

            wrong_manifest["model_artifact"] = minutes_model
            wrong_manifest["model_artifact_after_generation"] = minutes_model
            wrong_tokenizer = {
                **minutes_tokenizer,
                "sha256": "4" * 64,
            }
            wrong_manifest["tokenizer_artifact"] = wrong_tokenizer
            wrong_manifest[
                "tokenizer_artifact_after_generation"
            ] = wrong_tokenizer
            _write_json(wrong_manifest_path, wrong_manifest)
            with self.assertRaisesRegex(
                ValueError,
                "named frozen Minutes artifact",
            ):
                build_intervention_shard_manifest(
                    run_root=root,
                    generation_spec_file=spec,
                    prompt_manifest_file=prompt,
                    analysis_manifest_file=analysis,
                    projection_manifest_file=projection,
                    selected_indicators=["Indicator-A"],
                )

            wrong_manifest["tokenizer_artifact"] = minutes_tokenizer
            wrong_manifest[
                "tokenizer_artifact_after_generation"
            ] = wrong_tokenizer
            _write_json(wrong_manifest_path, wrong_manifest)
            with self.assertRaisesRegex(
                ValueError,
                "changed during generation",
            ):
                build_intervention_shard_manifest(
                    run_root=root,
                    generation_spec_file=spec,
                    prompt_manifest_file=prompt,
                    analysis_manifest_file=analysis,
                    projection_manifest_file=projection,
                    selected_indicators=["Indicator-A"],
                )

    def test_rejects_full_roster_as_a_partial_shard(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = root / "generation_spec.json"
            prompt = root / "prompt_manifest.json"
            analysis = root / "analysis_manifest.json"
            projection = root / "projection_manifest.json"
            _write_json(spec, {})
            _write_json(
                prompt,
                {
                    "baseline_indicator": "None",
                    "indicators": ["Indicator-A"],
                },
            )
            _write_json(analysis, {})
            _write_json(projection, {})
            with self.assertRaisesRegex(ValueError, "full-roster run"):
                build_intervention_shard_manifest(
                    run_root=root,
                    generation_spec_file=spec,
                    prompt_manifest_file=prompt,
                    analysis_manifest_file=analysis,
                    projection_manifest_file=projection,
                    selected_indicators=["Indicator-A"],
                )


if __name__ == "__main__":
    unittest.main()
