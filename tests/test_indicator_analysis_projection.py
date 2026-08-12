import json
import tempfile
import unittest
from pathlib import Path

from jobs.generation.project_indicator_analysis import (
    OUTPUT_FIELD,
    PROJECTION_POLICY,
    PROJECTION_ROW_SCHEMA_VERSION,
    PROJECTION_SCHEMA_VERSION,
    canonical_row_sha256,
    project_indicator_analysis,
)
from open_r1.provenance import sha256_file, sha256_text


class WhitespaceTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(text.split())))


TOKENIZER_ARTIFACT = {
    "path": "/test/frozen-minutes-tokenizer",
    "kind": "directory",
    "sha256": "a" * 64,
}


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
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _analysis_row(
    *,
    sample_id: str,
    meeting_date: str,
    indicator: str,
    generated: str,
) -> dict:
    return {
        "sample_id": sample_id,
        "meeting_date": meeting_date,
        "indicator": indicator,
        "generated": generated,
        "generated_sha256": sha256_text(generated),
        "generation_finish_reason": "stop",
    }


class TestIndicatorAnalysisProjection(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_path = self.root / "analysis" / "indicator_analysis.jsonl"
        self.analysis_manifest_path = (
            self.root / "analysis" / "analysis_manifest.json"
        )
        self.valid_rows = [
            _analysis_row(
                sample_id="2025-01-29::GDP-Growth",
                meeting_date="2025-01-29",
                indicator="GDP-Growth",
                generated=(
                    "  private reasoning for GDP  \n"
                    "</think>\n"
                    "  GDP growth was stable.  \n"
                ),
            ),
            _analysis_row(
                sample_id="2025-01-29::Unemployment-Rate",
                meeting_date="2025-01-29",
                indicator="Unemployment-Rate",
                generated=(
                    "private reasoning for unemployment\n"
                    "</think>\n"
                    "Labor-market conditions remained firm."
                ),
            ),
        ]
        self._write_source(self.valid_rows)

    def tearDown(self):
        self.temporary.cleanup()

    def _write_source(self, rows: list[dict]) -> None:
        _write_jsonl(self.input_path, rows)
        _write_json(
            self.analysis_manifest_path,
            {
                "schema_version": "indicator-analysis-generation-v2",
                "status": "complete",
                "output": {
                    "path": str(self.input_path.resolve()),
                    "sha256": sha256_file(self.input_path),
                    "row_count": len(rows),
                },
            },
        )

    def _project(self, output_name: str):
        return project_indicator_analysis(
            input_jsonl=self.input_path,
            analysis_manifest=self.analysis_manifest_path,
            tokenizer=WhitespaceTokenizer(),
            tokenizer_artifact=TOKENIZER_ARTIFACT,
            output_dir=self.root / output_name,
        )

    def test_projects_only_final_answers_with_complete_hash_bound_manifest(self):
        source_bytes_before = self.input_path.read_bytes()

        manifest = self._project("projection")

        output_path = self.root / "projection" / "minutes_analysis.jsonl"
        manifest_path = (
            self.root / "projection" / "analysis_projection_manifest.json"
        )
        projected_rows = _read_jsonl(output_path)
        disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest, disk_manifest)
        self.assertEqual(manifest["schema_version"], PROJECTION_SCHEMA_VERSION)
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse(manifest["generation_performed"])
        self.assertFalse(manifest["truncation_performed"])
        self.assertFalse(manifest["summarization_performed"])
        self.assertEqual(manifest["projection"]["policy"], PROJECTION_POLICY)
        self.assertEqual(manifest["projection"]["source_field"], "generated")
        self.assertEqual(manifest["projection"]["output_field"], OUTPUT_FIELD)
        self.assertEqual(manifest["projection"]["required_delimiter_count"], 1)
        self.assertFalse(manifest["projection"]["allow_plain_text_fallback"])

        self.assertEqual(self.input_path.read_bytes(), source_bytes_before)
        self.assertEqual(manifest["input"]["sha256"], sha256_file(self.input_path))
        self.assertEqual(
            manifest["input_analysis_manifest"]["sha256"],
            sha256_file(self.analysis_manifest_path),
        )
        self.assertEqual(manifest["output"]["sha256"], sha256_file(output_path))
        self.assertEqual(manifest["output"]["row_count"], 2)
        self.assertEqual(manifest["tokenizer_artifact"], TOKENIZER_ARTIFACT)

        self.assertEqual(
            [row[OUTPUT_FIELD] for row in projected_rows],
            [
                "GDP growth was stable.",
                "Labor-market conditions remained firm.",
            ],
        )
        projected_text = output_path.read_text(encoding="utf-8")
        self.assertNotIn("private reasoning", projected_text)
        self.assertNotIn("</think>", projected_text)
        self.assertTrue(all("generated" not in row for row in projected_rows))

        for raw, projected, inventory in zip(
            self.valid_rows,
            projected_rows,
            manifest["inventory"]["rows"],
            strict=True,
        ):
            answer = projected[OUTPUT_FIELD]
            self.assertEqual(
                projected["schema_version"],
                PROJECTION_ROW_SCHEMA_VERSION,
            )
            self.assertEqual(projected["source_row_sha256"], canonical_row_sha256(raw))
            self.assertEqual(
                projected["source_text_sha256"],
                sha256_text(raw["generated"]),
            )
            self.assertEqual(
                projected[f"{OUTPUT_FIELD}_sha256"],
                sha256_text(answer),
            )
            self.assertEqual(
                projected[f"{OUTPUT_FIELD}_token_count_no_special_tokens"],
                len(answer.split()),
            )
            self.assertEqual(
                inventory[f"{OUTPUT_FIELD}_sha256"],
                projected[f"{OUTPUT_FIELD}_sha256"],
            )

        self.assertEqual(manifest["inventory"]["row_count"], 2)
        self.assertEqual(manifest["inventory"]["meeting_count"], 1)
        self.assertEqual(manifest["inventory"]["indicator_count"], 2)
        self.assertEqual(manifest["inventory"]["minimum_final_answer_tokens"], 4)
        self.assertEqual(manifest["inventory"]["maximum_final_answer_tokens"], 4)

        second_manifest = self._project("projection-repeat")
        second_output = Path(second_manifest["output"]["path"])
        self.assertEqual(output_path.read_bytes(), second_output.read_bytes())
        self.assertEqual(
            manifest["output"]["sha256"],
            second_manifest["output"]["sha256"],
        )

    def test_rejects_missing_or_duplicate_think_delimiter(self):
        invalid_completions = {
            "missing": "reasoning followed by an unmarked final answer",
            "duplicate": (
                "reasoning</think>first answer</think>unexpected second answer"
            ),
        }
        for name, generated in invalid_completions.items():
            with self.subTest(name=name):
                row = dict(self.valid_rows[0])
                row["generated"] = generated
                row["generated_sha256"] = sha256_text(generated)
                self._write_source([row])
                output_dir = self.root / f"invalid-{name}"

                with self.assertRaisesRegex(ValueError, "exactly one"):
                    self._project(f"invalid-{name}")

                self.assertFalse(output_dir.exists())

    def test_rejects_empty_reasoning_or_final_answer(self):
        invalid_completions = {
            "empty-reasoning": "   \n</think>\nFinal answer.",
            "empty-answer": "Private reasoning.\n</think>\n   ",
        }
        for name, generated in invalid_completions.items():
            with self.subTest(name=name):
                row = dict(self.valid_rows[0])
                row["generated"] = generated
                row["generated_sha256"] = sha256_text(generated)
                self._write_source([row])
                output_dir = self.root / f"invalid-{name}"

                with self.assertRaisesRegex(
                    ValueError,
                    "non-empty reasoning and final answer",
                ):
                    self._project(f"invalid-{name}")

                self.assertFalse(output_dir.exists())

    def test_rejects_invalid_declared_generated_hash(self):
        row = dict(self.valid_rows[0])
        row["generated_sha256"] = "0" * 64
        self._write_source([row])

        with self.assertRaisesRegex(ValueError, "invalid generated_sha256"):
            self._project("bad-generated-hash")

        self.assertFalse((self.root / "bad-generated-hash").exists())

    def test_rejects_duplicate_sample_or_meeting_indicator_identity(self):
        duplicate_cases = {
            "sample-id": [
                self.valid_rows[0],
                {
                    **self.valid_rows[1],
                    "sample_id": self.valid_rows[0]["sample_id"],
                },
            ],
            "meeting-indicator": [
                self.valid_rows[0],
                {
                    **self.valid_rows[1],
                    "meeting_date": self.valid_rows[0]["meeting_date"],
                    "indicator": self.valid_rows[0]["indicator"],
                },
            ],
        }
        for name, rows in duplicate_cases.items():
            with self.subTest(name=name):
                self._write_source(rows)
                output_dir = self.root / f"duplicate-{name}"

                with self.assertRaisesRegex(ValueError, "Duplicate projection identity"):
                    self._project(f"duplicate-{name}")

                self.assertFalse(output_dir.exists())

    def test_refuses_to_overwrite_immutable_projection_directory(self):
        manifest = self._project("immutable")
        output_path = Path(manifest["output"]["path"])
        manifest_path = (
            self.root / "immutable" / "analysis_projection_manifest.json"
        )
        output_bytes = output_path.read_bytes()
        manifest_bytes = manifest_path.read_bytes()

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self._project("immutable")

        self.assertEqual(output_path.read_bytes(), output_bytes)
        self.assertEqual(manifest_path.read_bytes(), manifest_bytes)


if __name__ == "__main__":
    unittest.main()
