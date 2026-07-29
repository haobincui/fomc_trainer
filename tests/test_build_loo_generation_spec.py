import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from jobs.generation.build_loo_generation_spec import main
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    load_and_validate_generation_spec,
)


class TestBuildLooGenerationSpecCli(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path, Path, Path]:
        model = root / "model"
        tokenizer = root / "tokenizer"
        model.mkdir()
        tokenizer.mkdir()
        (model / "weights.bin").write_bytes(b"frozen weights")
        (tokenizer / "tokenizer.json").write_text(
            '{"frozen":true}\n',
            encoding="utf-8",
        )
        source = root / "prompts.jsonl"
        source.write_text('{"sample_id":"m1::views"}\n', encoding="utf-8")
        config = root / "generation_config.json"
        config.write_text(
            json.dumps(
                {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 128,
                    "context_limit": 512,
                }
            ),
            encoding="utf-8",
        )
        return model, tokenizer, source, config

    def _arguments(
        self,
        root: Path,
        *,
        model_declarations: list[str] | None = None,
        config: Path | None = None,
    ) -> tuple[list[str], Path, tuple[Path, Path, Path]]:
        model, tokenizer, source, default_config = self._inputs(root)
        output = root / "run" / "generation_spec.json"
        arguments = [
            "--run-id",
            "pilot-20260728",
            "--phase",
            "seven-indicator-pilot",
            "--population-id",
            "evaluation-13",
            "--output",
            str(output),
        ]
        for declaration in model_declarations or [f"minutes={model}"]:
            arguments.extend(["--model", declaration])
        arguments.extend(
            [
                "--tokenizer",
                f"minutes={tokenizer}",
                "--source",
                f"prompts={source}",
                "--generation-config",
                str(config or default_config),
                "--replicate-seed",
                "20260728",
            ]
        )
        return arguments, output, (model, tokenizer, source)

    def test_cli_writes_valid_spec_and_prints_both_digests(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            arguments, output, artifacts = self._arguments(root)
            before = {
                path: (
                    {
                        candidate.relative_to(path): candidate.read_bytes()
                        for candidate in path.rglob("*")
                        if candidate.is_file()
                    }
                    if path.is_dir()
                    else path.read_bytes()
                )
                for path in artifacts
            }
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(arguments)

            spec = load_and_validate_generation_spec(output)
            self.assertEqual(exit_code, 0)
            self.assertEqual(spec["run_id"], "pilot-20260728")
            self.assertIn(f"output={output.resolve()}", stdout.getvalue())
            self.assertIn(
                f"file_sha256={sha256_file(output)}",
                stdout.getvalue(),
            )
            self.assertIn(
                f"payload_sha256={spec['integrity']['payload_sha256']}",
                stdout.getvalue(),
            )
            after = {
                path: (
                    {
                        candidate.relative_to(path): candidate.read_bytes()
                        for candidate in path.rglob("*")
                        if candidate.is_file()
                    }
                    if path.is_dir()
                    else path.read_bytes()
                )
                for path in artifacts
            }
            self.assertEqual(before, after)

    def test_cli_rejects_duplicate_artifact_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = root / "duplicate-model"
            model.mkdir()
            (model / "weights.bin").write_bytes(b"other frozen weights")
            arguments, _, _ = self._arguments(
                root,
                model_declarations=[
                    f"minutes={root / 'model'}",
                    f"minutes={model}",
                ],
            )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(SystemExit, "2"):
                    main(arguments)

            self.assertIn("Duplicate --model name", stderr.getvalue())

    def test_cli_rejects_malformed_named_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            arguments, _, _ = self._arguments(
                root,
                model_declarations=["missing-equals-sign"],
            )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(SystemExit, "2"):
                    main(arguments)

            self.assertIn("Expected NAME=PATH", stderr.getvalue())

    def test_cli_rejects_non_object_generation_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            invalid_config = root / "invalid_config.json"
            invalid_config.write_text("[]\n", encoding="utf-8")
            arguments, output, _ = self._arguments(
                root,
                config=invalid_config,
            )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(SystemExit, "2"):
                    main(arguments)

            self.assertIn("non-empty JSON object", stderr.getvalue())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
