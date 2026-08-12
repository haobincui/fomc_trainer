import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from jobs.generation.finalize_loo_generation import (
    RELEASE_SCHEMA_VERSION,
    _validate_analysis,
    finalize_loo_generation,
    main,
)
from jobs.generation.canonical_indicator_analysis import (
    SYSTEM_PROMPT_VERSION,
    build_indicator_analysis_system_prompt,
)
from jobs.generation.project_indicator_analysis import project_indicator_analysis
from open_r1.generate import get_system_prompt
from open_r1.provenance import (
    fingerprint_artifact_path,
    sha256_file,
    sha256_text,
)
from open_r1.validator.loo_generation_spec import (
    build_generation_spec,
    derive_row_seed,
    seal_manifest,
    validate_manifest_integrity,
    write_frozen_generation_spec,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


class WhitespaceTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(text.split())))


class SyntheticCanonicalRun:
    def __init__(
        self,
        root: Path,
        *,
        mismatch_neutral_baseline: bool = False,
        length_finish_run: str | None = None,
    ):
        self.root = root
        self.population_id = "formal-test"
        self.context = "formal_test"
        self.baseline = "None"
        self.indicator = "GDP-Growth"
        self.sample_id = "2025-01-29::economic_situation"
        self.model = root / "inputs" / "model"
        self.tokenizer = root / "inputs" / "tokenizer"
        self.model.mkdir(parents=True)
        self.tokenizer.mkdir(parents=True)
        (self.model / "weights.bin").write_bytes(b"frozen model")
        (self.tokenizer / "tokenizer.json").write_bytes(b"frozen tokenizer")

        self.analysis_output = root / "analysis" / "indicator_analysis.jsonl"
        _write_jsonl(
            self.analysis_output,
            [
                {
                    "sample_id": "2025-01-29::GDP-Growth",
                    "meeting_date": "2025-01-29",
                    "indicator": self.indicator,
                    "generated": "Private reasoning.</think>GDP was stable.",
                    "generated_sha256": sha256_text(
                        "Private reasoning.</think>GDP was stable."
                    ),
                }
            ],
        )
        self.analysis_manifest = root / "analysis" / "analysis_manifest.json"
        _write_json(
            self.analysis_manifest,
            {
                "schema_version": "indicator-analysis-generation-v1",
                "status": "complete",
                "run_id": "analysis-run",
                "generation_only": True,
                "training_performed": False,
                "inputs": {
                    "model": fingerprint_artifact_path(self.model),
                    "tokenizer": fingerprint_artifact_path(self.tokenizer),
                },
                "inventory": {
                    "row_count": 1,
                    "population_id": self.population_id,
                    "rows": [{"sample_id": "2025-01-29::GDP-Growth"}],
                },
                "output": {
                    "path": str(self.analysis_output.resolve()),
                    "sha256": sha256_file(self.analysis_output),
                },
            },
        )
        self.projection_root = root / "analysis_projection"
        project_indicator_analysis(
            input_jsonl=self.analysis_output,
            analysis_manifest=self.analysis_manifest,
            tokenizer=WhitespaceTokenizer(),
            tokenizer_artifact=fingerprint_artifact_path(self.tokenizer),
            output_dir=self.projection_root,
        )
        self.projection_output = self.projection_root / "minutes_analysis.jsonl"
        self.projection_manifest = (
            self.projection_root / "analysis_projection_manifest.json"
        )

        self.prompt_root = root / "prompts"
        self.prompt_inventory = self._write_prompts()
        self.prompt_manifest = self.prompt_root / "prompt_manifest.json"
        _write_json(
            self.prompt_manifest,
            {
                "schema_version": "loo-prompt-manifest-v2",
                "population_id": self.population_id,
                "population_context": self.context,
                "population_dates": ["2025-01-29"],
                "section_families": ["economic_situation"],
                "baseline_indicator": self.baseline,
                "indicators": [self.indicator],
                "analysis_text_field": "minutes_analysis",
                "context_budget": {
                    "policy": "exact-chat-template-no-truncation-v1",
                    "system_prompt_sha256": sha256_text(get_system_prompt()),
                    "token_count_policy": (
                        "tokenizer.apply_chat_template(tokenize=True,"
                        "add_generation_prompt=True)"
                    ),
                    "max_new_tokens": 8192,
                    "max_model_len": 16384,
                    "audited_prompt_row_count": 4,
                    "maximum_chat_prompt_tokens": 10,
                    "minimum_context_headroom_tokens": 8182,
                    "all_rows_fit": True,
                    "input_truncation": "forbidden",
                },
                "counts": {
                    "meeting_count": 1,
                    "section_count": 1,
                    "unit_count": 1,
                    "indicator_count": 1,
                    "analysis_block_count": 1,
                    "full_prompt_count": 1,
                    "delete_prompt_count": 1,
                    "neutral_prompt_count": 1,
                },
                "source_artifacts": [
                    {
                        "role": "analysis_blocks",
                        "path": str(self.projection_output.resolve()),
                        "sha256": sha256_file(self.projection_output),
                    }
                ],
                "artifacts": [
                    {
                        "relative_path": relative_path,
                        "arm": arm,
                        "indicator": indicator,
                        "context": self.context,
                        "row_count": 1,
                        "sha256": sha256_file(path),
                    }
                    for (
                        folder,
                        indicator,
                    ), (path, relative_path, arm) in self.prompt_inventory.items()
                ],
            },
        )

        self.spec_path = root / "generation_spec.json"
        spec = build_generation_spec(
            run_id="canonical-test-run",
            phase="formal",
            population_id=self.population_id,
            models={"minutes_model": self.model},
            tokenizers={"minutes_tokenizer": self.tokenizer},
            sources={
                "analysis_output": self.analysis_output,
                "analysis_manifest": self.analysis_manifest,
                "analysis_projection_output": self.projection_output,
                "analysis_projection_manifest": self.projection_manifest,
                "prompt_manifest": self.prompt_manifest,
                "exact_delete_prompts": self.prompt_root / "exact_delete",
                "neutral_prompts": self.prompt_root / "neutral",
            },
            generation_config={
                "generation_only": True,
                "training_performed": False,
                "execution": {
                    "gpu_policy": "single-physical-gpu-by-nvidia-smi-index-v1",
                    "physical_gpu_index": 1,
                    "expected_visible_cuda_devices": 1,
                    "tensor_parallel_size": 1,
                },
                "analysis_projection": {
                    "policy": "deepseek-final-answer-after-think-v1",
                    "source_field": "generated",
                    "output_field": "minutes_analysis",
                    "required_format": "deepseek_think_completion",
                    "delimiter": "</think>",
                    "required_delimiter_count": 1,
                    "allow_plain_text_fallback": False,
                    "require_nonempty_reasoning": True,
                    "require_nonempty_final_answer": True,
                    "truncation": "forbidden",
                    "secondary_generation": "forbidden",
                },
                "decoding": {
                    "minutes_primary": {
                        "max_new_tokens": 8192,
                        "max_model_len": 16384,
                    },
                    "minutes_stochastic_robustness": {
                        "max_new_tokens": 8192,
                        "max_model_len": 16384,
                    },
                },
                "sections": [
                    {
                        "section_id": "economic_situation",
                        "section_name": "economic_situation",
                    }
                ],
                "populations": {
                    self.population_id: {
                        "phase": "formal",
                        "meeting_dates": ["2025-01-29"],
                    }
                },
                "invariants": {
                    "indicator_count_per_meeting": 1,
                    "section_count_per_meeting": 1,
                },
            },
            replicate_seeds=[
                20260728,
                20260729,
                21260729,
                22260729,
                23260729,
                24260729,
            ],
        )
        self.spec_sha256 = write_frozen_generation_spec(self.spec_path, spec)
        self.spec = spec
        self.output_paths: dict[str, list[Path]] = {}

        run_settings = {
            "deletion_primary": (
                "exact_delete",
                "indicator_block_deletion",
                1,
                20260728,
                0.0,
                1.0,
            ),
            "neutral_primary": (
                "neutral",
                "indicator_block_neutral_replacement",
                1,
                20260728,
                0.0,
                1.0,
            ),
            "deletion_stochastic": (
                "exact_delete",
                "indicator_block_deletion",
                5,
                20260729,
                0.6,
                0.9,
            ),
            "neutral_stochastic": (
                "neutral",
                "indicator_block_neutral_replacement",
                5,
                20260729,
                0.6,
                0.9,
            ),
        }
        for name, settings in run_settings.items():
            self._write_generation_run(
                name,
                *settings,
                mismatch_neutral_baseline=(
                    mismatch_neutral_baseline and name == "neutral_primary"
                ),
                length_finish=name == length_finish_run,
            )

    def _write_prompts(self):
        inventory = {}
        declarations = (
            ("exact_delete", self.baseline, "full", "full prompt"),
            ("exact_delete", self.indicator, "delete", "delete prompt"),
            ("neutral", self.baseline, "full_export", "full prompt"),
            ("neutral", self.indicator, "neutral", "neutral prompt"),
        )
        for folder, indicator, arm, prompt in declarations:
            relative_path = f"{folder}/{indicator}_masked_{self.context}.jsonl"
            path = self.prompt_root / relative_path
            _write_jsonl(
                path,
                [
                    {
                        "sample_id": self.sample_id,
                        "meeting_date": "2025-01-29",
                        "section_name": "economic_situation",
                        "indicator": indicator,
                        "arm": arm,
                        "prompt": prompt,
                        "prompt_sha256": sha256_text(prompt),
                        "prompt_token_count_chat_template": 10,
                        "context_headroom_tokens": 8182,
                    }
                ],
            )
            inventory[(folder, indicator)] = (
                path,
                relative_path,
                arm,
            )
        return inventory

    def _write_generation_run(
        self,
        name: str,
        prompt_folder: str,
        strategy: str,
        replicate_count: int,
        base_seed: int,
        temperature: float,
        top_p: float,
        *,
        mismatch_neutral_baseline: bool,
        length_finish: bool,
    ) -> None:
        directory = self.root / "generations" / name
        directory.mkdir(parents=True)
        intervention = directory / "intervention_manifest.json"
        _write_json(intervention, {"strategy": strategy})
        model_fingerprint = fingerprint_artifact_path(self.model)
        tokenizer_fingerprint = fingerprint_artifact_path(self.tokenizer)
        replicate_seeds = [
            base_seed + index * 1_000_000 for index in range(replicate_count)
        ]
        input_artifacts = []
        expected_artifacts = []
        paths = []
        for indicator in (self.baseline, self.indicator):
            prompt_path, _, _ = self.prompt_inventory[(prompt_folder, indicator)]
            prompt_row = json.loads(prompt_path.read_text(encoding="utf-8"))
            input_artifacts.append(
                {
                    "path": str(prompt_path.resolve()),
                    "indicator": indicator,
                    "context": self.context,
                    "source_row_count": 1,
                    "source_file_sha256": sha256_file(prompt_path),
                }
            )
            for replicate_id, replicate_seed in enumerate(replicate_seeds):
                output_path = directory / (
                    f"{indicator}_masked_{self.context}_{replicate_id}.jsonl"
                )
                generated = (
                    f"full::{name.split('_')[-1]}::{replicate_id}"
                    if indicator == self.baseline
                    else f"treated::{name}::{replicate_id}"
                )
                if mismatch_neutral_baseline and indicator == self.baseline:
                    generated += "::mismatch"
                finish_reason = "length" if length_finish else "stop"
                row = {
                    **prompt_row,
                    "generated": generated,
                    "generated_sha256": sha256_text(generated),
                    "source_prompt_sha256": sha256_text(prompt_row["prompt"]),
                    "replicate_id": str(replicate_id),
                    "generation_seed": derive_row_seed(
                        replicate_seed,
                        self.sample_id,
                    ),
                    "generation_seed_policy": "sample-id-sha256-v1",
                    "masking_strategy": strategy,
                    "evaluation_context": self.context,
                    "decoding_temperature": temperature,
                    "decoding_top_p": top_p,
                    "max_new_tokens": 8192,
                    "max_model_len": 16384,
                    "generation_model_sha256": model_fingerprint["sha256"],
                    "generation_tokenizer_sha256": tokenizer_fingerprint["sha256"],
                    "generation_system_prompt_sha256": sha256_text(
                        get_system_prompt()
                    ),
                    "prompt_preflight_token_count": 10,
                    "prompt_token_count": 10,
                    "output_token_count": 4,
                    "generation_finish_reason": finish_reason,
                    "input_was_truncated": False,
                }
                _write_jsonl(output_path, [row])
                paths.append(output_path)
                expected_artifacts.append(
                    {
                        "path": str(output_path.resolve()),
                        "relative_path": output_path.name,
                        "indicator": indicator,
                        "context": self.context,
                        "replicate_id": str(replicate_id),
                        "replicate_seed": replicate_seed,
                        "source_file_sha256": sha256_file(prompt_path),
                        "source_row_count": 1,
                        "output_sha256": sha256_file(output_path),
                    }
                )
        self.output_paths[name] = paths
        manifest = {
            "schema_version": "loo-generation-v4",
            "input_folder": str((self.prompt_root / prompt_folder).resolve()),
            "output_dir": str(directory.resolve()),
            "model_artifact": model_fingerprint,
            "model_artifact_after_generation": model_fingerprint,
            "tokenizer_artifact": tokenizer_fingerprint,
            "tokenizer_artifact_after_generation": tokenizer_fingerprint,
            "simulation_step": replicate_count,
            "base_seed": base_seed,
            "replicate_seeds": replicate_seeds,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": 8192,
            "max_model_len": 16384,
            "seed_policy": "sample-id-sha256-v1",
            "require_normal_finish": True,
            "system_prompt_sha256": sha256_text(get_system_prompt()),
            "execution_environment": {
                "physical_gpu_index": "1",
                "physical_gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001",
                "cuda_visible_devices": "1",
                "declared_visible_device_count": "1",
                "tensor_parallel_size": 1,
            },
            "masking_strategy": strategy,
            "intervention_manifest": {
                "relative_path": intervention.name,
                "sha256": sha256_file(intervention),
            },
            "prompt_manifest": {
                "path": str(self.prompt_manifest.resolve()),
                "sha256": sha256_file(self.prompt_manifest),
                "population_id": self.population_id,
            },
            "generation_spec": {
                "path": str(self.spec_path.resolve()),
                "file_sha256": self.spec_sha256,
                "payload_sha256": self.spec["integrity"]["payload_sha256"],
                "run_id": self.spec["run_id"],
            },
            "input_artifacts": input_artifacts,
            "expected_artifacts": expected_artifacts,
        }
        _write_json(directory / "generation_manifest.json", manifest)


class TestFinalizeLooGeneration(unittest.TestCase):
    def _write_v2_analysis_fixture(self, root: Path):
        model = root / "inputs" / "analysis-model"
        tokenizer = root / "inputs" / "analysis-tokenizer"
        model.mkdir(parents=True)
        tokenizer.mkdir(parents=True)
        (model / "weights.bin").write_bytes(b"model")
        (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
        sample_id = "2025-01-29::GDP-Growth"
        hard_max_new_tokens = 256
        requested_max_output_tokens = 128
        system_prompt_sha256 = sha256_text(
            build_indicator_analysis_system_prompt(requested_max_output_tokens)
        )
        error = {
            "sample_id": sample_id,
            "error_type": "token_limit_finish",
            "message": "Generation ended at a token limit",
            "finish_reason": "length",
            "input_token_count": 100,
            "output_token_count": 256,
            "max_new_tokens": 256,
        }
        output = root / "analysis" / "indicator_analysis.jsonl"
        _write_jsonl(
            output,
            [
                {
                    "sample_id": sample_id,
                    "generated": "Truncated but non-empty analysis.",
                    "generation_validation_status": ("accepted_token_limit_error"),
                    "generation_validation_error": error,
                    "generation_finish_reason": "length",
                    "input_was_truncated": False,
                    "generation_system_prompt_sha256": (system_prompt_sha256),
                    "generation_requested_max_output_tokens": (
                        requested_max_output_tokens
                    ),
                    "max_new_tokens": hard_max_new_tokens,
                }
            ],
        )
        manifest_path = root / "analysis" / "analysis_manifest.json"
        manifest = {
            "schema_version": "indicator-analysis-generation-v2",
            "status": "complete",
            "run_id": "analysis-v2",
            "generation_only": True,
            "training_performed": False,
            "inputs": {
                "model": fingerprint_artifact_path(model),
                "tokenizer": fingerprint_artifact_path(tokenizer),
            },
            "system_prompt": {
                "version": SYSTEM_PROMPT_VERSION,
                "sha256": system_prompt_sha256,
                "requested_max_output_tokens": (requested_max_output_tokens),
                "hard_max_new_tokens": hard_max_new_tokens,
            },
            "completion_validation": {
                "policy": "bounded-token-limit-errors-v1",
                "max_token_limit_errors": 2,
                "observed_token_limit_error_count": 1,
                "passed_count": 0,
                "not_evaluated_count": 0,
                "errors": [error],
            },
            "inventory": {
                "row_count": 1,
                "population_id": "pilot",
                "rows": [{"sample_id": sample_id}],
            },
            "output": {
                "path": str(output.resolve()),
                "sha256": sha256_file(output),
            },
        }
        _write_json(manifest_path, manifest)
        spec = {
            "generation_config": {
                "decoding": {
                    "indicator_analysis": {
                        "max_new_tokens": hard_max_new_tokens,
                        "requested_max_output_tokens": (requested_max_output_tokens),
                        "max_token_limit_errors": 2,
                    }
                },
                "invariants": {
                    "indicator_analysis_token_limit_finish": {
                        "policy": "bounded-token-limit-errors-v1",
                        "max_errors": 2,
                    }
                },
            },
            "frozen_artifacts": {
                "models": {
                    "analysis_model": fingerprint_artifact_path(model),
                },
                "tokenizers": {
                    "analysis_tokenizer": fingerprint_artifact_path(tokenizer),
                },
                "sources": {
                    "analysis_output": {
                        "sha256": sha256_file(output),
                    },
                    "analysis_manifest": {
                        "sha256": sha256_file(manifest_path),
                    },
                },
            },
        }
        return manifest_path, manifest, spec

    def test_accepts_and_reports_v2_bounded_analysis_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path, _, spec = self._write_v2_analysis_fixture(root)

            summary = _validate_analysis(
                manifest_path=manifest_path,
                run_root=root,
                spec=spec,
            )

            validation = summary["completion_validation"]
            self.assertEqual(
                validation["observed_token_limit_error_count"],
                1,
            )
            self.assertEqual(validation["max_token_limit_errors"], 2)
            self.assertEqual(
                summary["system_prompt"]["requested_max_output_tokens"],
                128,
            )

    def test_rejects_v2_analysis_error_count_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path, manifest, spec = self._write_v2_analysis_fixture(root)
            manifest["completion_validation"]["observed_token_limit_error_count"] = 0
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError,
                "summary differs from its rows",
            ):
                _validate_analysis(
                    manifest_path=manifest_path,
                    run_root=root,
                    spec=spec,
                )

    def test_rejects_v2_analysis_system_prompt_limit_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path, manifest, spec = self._write_v2_analysis_fixture(root)
            manifest["system_prompt"]["requested_max_output_tokens"] = 129
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError,
                "system prompt or output limits differ",
            ):
                _validate_analysis(
                    manifest_path=manifest_path,
                    run_root=root,
                    spec=spec,
                )

    def test_cli_writes_sealed_generation_only_release(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = SyntheticCanonicalRun(Path(temporary_directory))
            output = run.root / "release.json"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--run-root",
                        str(run.root),
                        "--generation-spec",
                        str(run.spec_path),
                        "--generation-spec-sha256",
                        run.spec_sha256,
                        "--analysis-manifest",
                        str(run.analysis_manifest),
                        "--projection-manifest",
                        str(run.projection_manifest),
                        "--prompt-manifest",
                        str(run.prompt_manifest),
                        "--output",
                        str(output),
                    ]
                )

            release = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                release["schema_version"],
                RELEASE_SCHEMA_VERSION,
            )
            self.assertTrue(release["generation_only"])
            self.assertFalse(release["training_performed"])
            self.assertFalse(release["scoring_performed"])
            self.assertEqual(
                release["baseline_equivalence"]["primary"]["status"],
                "exact_match",
            )
            self.assertEqual(
                release["generation_runs"]["deletion_stochastic"]["replicate_count"],
                5,
            )
            validate_manifest_integrity(release)
            self.assertIn("status=complete", stdout.getvalue())
            self.assertIn(f"file_sha256={sha256_file(output)}", stdout.getvalue())

    def test_rejects_cross_strategy_full_baseline_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = SyntheticCanonicalRun(
                Path(temporary_directory),
                mismatch_neutral_baseline=True,
            )
            with self.assertRaisesRegex(ValueError, "full-baseline mismatch"):
                finalize_loo_generation(
                    run_root=run.root,
                    generation_spec=run.spec_path,
                    generation_spec_sha256=run.spec_sha256,
                    analysis_manifest=run.analysis_manifest,
                    projection_manifest=run.projection_manifest,
                    prompt_manifest=run.prompt_manifest,
                    output=run.root / "release.json",
                )

    def test_rejects_length_finish(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = SyntheticCanonicalRun(
                Path(temporary_directory),
                length_finish_run="deletion_primary",
            )
            with self.assertRaisesRegex(ValueError, "token limit"):
                finalize_loo_generation(
                    run_root=run.root,
                    generation_spec=run.spec_path,
                    generation_spec_sha256=run.spec_sha256,
                    analysis_manifest=run.analysis_manifest,
                    projection_manifest=run.projection_manifest,
                    prompt_manifest=run.prompt_manifest,
                    output=run.root / "release.json",
                )

    def test_rejects_output_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = SyntheticCanonicalRun(Path(temporary_directory))
            tampered = run.output_paths["deletion_primary"][0]
            with tampered.open("a", encoding="utf-8") as handle:
                handle.write('{"tampered":true}\n')

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                finalize_loo_generation(
                    run_root=run.root,
                    generation_spec=run.spec_path,
                    generation_spec_sha256=run.spec_sha256,
                    analysis_manifest=run.analysis_manifest,
                    projection_manifest=run.projection_manifest,
                    prompt_manifest=run.prompt_manifest,
                    output=run.root / "release.json",
                )

    def test_rejects_unprespecified_replicate_seed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = SyntheticCanonicalRun(Path(temporary_directory))
            manifest_path = (
                run.root
                / "generations"
                / "deletion_primary"
                / "generation_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["replicate_seeds"] = [7]
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(ValueError, "frozen design"):
                finalize_loo_generation(
                    run_root=run.root,
                    generation_spec=run.spec_path,
                    generation_spec_sha256=run.spec_sha256,
                    analysis_manifest=run.analysis_manifest,
                    projection_manifest=run.projection_manifest,
                    prompt_manifest=run.prompt_manifest,
                    output=run.root / "release.json",
                )

    def test_rejects_minutes_context_different_from_frozen_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = SyntheticCanonicalRun(Path(temporary_directory))
            manifest_path = (
                run.root
                / "generations"
                / "deletion_primary"
                / "generation_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["max_model_len"] = 32768
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError,
                "generation configuration mismatch",
            ):
                finalize_loo_generation(
                    run_root=run.root,
                    generation_spec=run.spec_path,
                    generation_spec_sha256=run.spec_sha256,
                    analysis_manifest=run.analysis_manifest,
                    projection_manifest=run.projection_manifest,
                    prompt_manifest=run.prompt_manifest,
                    output=run.root / "release.json",
                )

    def test_rejects_generation_not_pinned_to_physical_gpu1(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = SyntheticCanonicalRun(Path(temporary_directory))
            manifest_path = (
                run.root
                / "generations"
                / "deletion_primary"
                / "generation_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["execution_environment"]["physical_gpu_index"] = "0"
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(ValueError, "physical GPU 1"):
                finalize_loo_generation(
                    run_root=run.root,
                    generation_spec=run.spec_path,
                    generation_spec_sha256=run.spec_sha256,
                    analysis_manifest=run.analysis_manifest,
                    projection_manifest=run.projection_manifest,
                    prompt_manifest=run.prompt_manifest,
                    output=run.root / "release.json",
                )

    def test_rejects_model_bound_only_under_non_minutes_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = SyntheticCanonicalRun(Path(temporary_directory))
            rogue_model = run.root / "inputs" / "analysis-only-model"
            rogue_model.mkdir()
            (rogue_model / "weights.bin").write_bytes(b"analysis-only")
            rogue_fingerprint = fingerprint_artifact_path(rogue_model)

            resealed_payload = {
                key: value
                for key, value in run.spec.items()
                if key != "integrity"
            }
            resealed_payload["frozen_artifacts"]["models"][
                "analysis_model"
            ] = rogue_fingerprint
            run.spec = seal_manifest(resealed_payload)
            _write_json(run.spec_path, run.spec)
            run.spec_sha256 = sha256_file(run.spec_path)

            for run_name in (
                "deletion_primary",
                "neutral_primary",
                "deletion_stochastic",
                "neutral_stochastic",
            ):
                manifest_path = (
                    run.root
                    / "generations"
                    / run_name
                    / "generation_manifest.json"
                )
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                manifest["generation_spec"]["file_sha256"] = run.spec_sha256
                manifest["generation_spec"]["payload_sha256"] = run.spec[
                    "integrity"
                ]["payload_sha256"]
                if run_name == "deletion_primary":
                    manifest["model_artifact"] = rogue_fingerprint
                    manifest[
                        "model_artifact_after_generation"
                    ] = rogue_fingerprint
                _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError,
                "models.minutes_model",
            ):
                finalize_loo_generation(
                    run_root=run.root,
                    generation_spec=run.spec_path,
                    generation_spec_sha256=run.spec_sha256,
                    analysis_manifest=run.analysis_manifest,
                    projection_manifest=run.projection_manifest,
                    prompt_manifest=run.prompt_manifest,
                    output=run.root / "release.json",
                )

    def test_rejects_tokenizer_bound_only_under_non_minutes_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = SyntheticCanonicalRun(Path(temporary_directory))
            rogue_tokenizer = run.root / "inputs" / "analysis-only-tokenizer"
            rogue_tokenizer.mkdir()
            (rogue_tokenizer / "tokenizer.json").write_bytes(
                b"analysis-only"
            )
            rogue_fingerprint = fingerprint_artifact_path(rogue_tokenizer)

            resealed_payload = {
                key: value
                for key, value in run.spec.items()
                if key != "integrity"
            }
            resealed_payload["frozen_artifacts"]["tokenizers"][
                "analysis_tokenizer"
            ] = rogue_fingerprint
            run.spec = seal_manifest(resealed_payload)
            _write_json(run.spec_path, run.spec)
            run.spec_sha256 = sha256_file(run.spec_path)

            for run_name in (
                "deletion_primary",
                "neutral_primary",
                "deletion_stochastic",
                "neutral_stochastic",
            ):
                manifest_path = (
                    run.root
                    / "generations"
                    / run_name
                    / "generation_manifest.json"
                )
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                manifest["generation_spec"]["file_sha256"] = run.spec_sha256
                manifest["generation_spec"]["payload_sha256"] = run.spec[
                    "integrity"
                ]["payload_sha256"]
                if run_name == "deletion_primary":
                    manifest["tokenizer_artifact"] = rogue_fingerprint
                    manifest[
                        "tokenizer_artifact_after_generation"
                    ] = rogue_fingerprint
                _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError,
                "tokenizers.minutes_tokenizer",
            ):
                finalize_loo_generation(
                    run_root=run.root,
                    generation_spec=run.spec_path,
                    generation_spec_sha256=run.spec_sha256,
                    analysis_manifest=run.analysis_manifest,
                    projection_manifest=run.projection_manifest,
                    prompt_manifest=run.prompt_manifest,
                    output=run.root / "release.json",
                )

    def test_requires_external_generation_spec_hash(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = SyntheticCanonicalRun(Path(temporary_directory))
            with self.assertRaisesRegex(ValueError, "externally frozen file"):
                finalize_loo_generation(
                    run_root=run.root,
                    generation_spec=run.spec_path,
                    generation_spec_sha256="f" * 64,
                    analysis_manifest=run.analysis_manifest,
                    projection_manifest=run.projection_manifest,
                    prompt_manifest=run.prompt_manifest,
                    output=run.root / "release.json",
                )

    def test_rejects_sample_filtered_smoke_as_population_release(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = SyntheticCanonicalRun(Path(temporary_directory))
            manifest_path = (
                run.root
                / "generations"
                / "deletion_primary"
                / "generation_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sample_selection"] = {
                "policy": "explicit-sample-id-filter-v1",
                "mode": "filtered_smoke",
                "requested_sample_ids": [run.sample_id],
                "selected_sample_ids": [run.sample_id],
                "selected_row_count_per_artifact": 1,
                "full_source_row_count_per_artifact": 1,
                "population_release_allowed": False,
            }
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError,
                "sample-filtered smoke output",
            ):
                finalize_loo_generation(
                    run_root=run.root,
                    generation_spec=run.spec_path,
                    generation_spec_sha256=run.spec_sha256,
                    analysis_manifest=run.analysis_manifest,
                    projection_manifest=run.projection_manifest,
                    prompt_manifest=run.prompt_manifest,
                    output=run.root / "release.json",
                )


if __name__ == "__main__":
    unittest.main()
