import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from jobs.generation.finalize_loo_generation import (
    RELEASE_SCHEMA_VERSION,
    finalize_loo_generation,
    main,
)
from open_r1.provenance import (
    fingerprint_artifact_path,
    sha256_file,
    sha256_text,
)
from open_r1.validator.loo_generation_spec import (
    build_generation_spec,
    derive_row_seed,
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
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


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
                    "generated": "GDP was stable.",
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

        self.prompt_root = root / "prompts"
        self.prompt_inventory = self._write_prompts()
        self.prompt_manifest = self.prompt_root / "prompt_manifest.json"
        _write_json(
            self.prompt_manifest,
            {
                "schema_version": "loo-prompt-manifest-v1",
                "population_id": self.population_id,
                "population_context": self.context,
                "population_dates": ["2025-01-29"],
                "section_families": ["economic_situation"],
                "baseline_indicator": self.baseline,
                "indicators": [self.indicator],
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
                        "path": str(self.analysis_output.resolve()),
                        "sha256": sha256_file(self.analysis_output),
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
                "prompt_manifest": self.prompt_manifest,
                "exact_delete_prompts": self.prompt_root / "exact_delete",
                "neutral_prompts": self.prompt_root / "neutral",
            },
            generation_config={
                "generation_only": True,
                "training_performed": False,
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
            relative_path = (
                f"{folder}/{indicator}_masked_{self.context}.jsonl"
            )
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
            base_seed + index * 1_000_000
            for index in range(replicate_count)
        ]
        input_artifacts = []
        expected_artifacts = []
        paths = []
        for indicator in (self.baseline, self.indicator):
            prompt_path, _, _ = self.prompt_inventory[
                (prompt_folder, indicator)
            ]
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
                    "max_new_tokens": 20,
                    "max_model_len": 100,
                    "generation_model_sha256": model_fingerprint["sha256"],
                    "generation_tokenizer_sha256": tokenizer_fingerprint["sha256"],
                    "generation_system_prompt_sha256": "a" * 64,
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
            "max_new_tokens": 20,
            "max_model_len": 100,
            "seed_policy": "sample-id-sha256-v1",
            "require_normal_finish": True,
            "system_prompt_sha256": "a" * 64,
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
                release["generation_runs"]["deletion_stochastic"][
                    "replicate_count"
                ],
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
                    prompt_manifest=run.prompt_manifest,
                    output=run.root / "release.json",
                )


if __name__ == "__main__":
    unittest.main()
