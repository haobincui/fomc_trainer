import json
import tempfile
import unittest
from pathlib import Path

from jobs.generation.loo_prompt_builder import build_canonical_loo_prompts
from jobs.generation.mask_generation import build_intervention_manifest
from open_r1.validator.intervention import (
    has_line_block_boundaries,
    single_contiguous_deletion,
)


class WhitespaceTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(text.split())))


TOKENIZER_ARTIFACT = {
    "path": "/test/frozen-tokenizer",
    "kind": "directory",
    "sha256": "a" * 64,
}


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestCanonicalLooPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.indicators = [f"Indicator-{index:02d}" for index in range(26)]
        self.dates = ["2024-01-31", "2024-03-20"]
        self.sections = ["Participants' Views", "Staff Economic Situation"]
        self.roster_path = self.root / "indicator_roster.json"
        self.population_path = self.root / "population.json"
        self.section_path = self.root / "sections.json"
        self.blocks_path = self.root / "blocks.jsonl"

        _write_json(
            self.roster_path,
            {
                "schema_version": "loo-intervention-roster-v1",
                "roster_id": "test-26-v1",
                "baseline_indicator": "None",
                "contexts": ["source-context"],
                "indicators": self.indicators,
                "indicator_markers": {
                    indicator: [indicator.replace("-", " ")]
                    for indicator in self.indicators
                },
            },
        )
        _write_json(
            self.population_path,
            {
                "population_id": "formal-recent-2-v1",
                "dates": self.dates,
            },
        )
        _write_json(
            self.section_path,
            {
                "roster_id": "test-sections-v1",
                "sections": self.sections,
            },
        )
        self.valid_rows = [
            {
                "meeting_date": meeting_date,
                "indicator": indicator,
                "generated": (
                    f"private analysis evidence for {meeting_date} and "
                    f"{indicator} with several neutralizable words"
                ),
            }
            for meeting_date in self.dates
            for indicator in self.indicators
        ]
        _write_jsonl(self.blocks_path, self.valid_rows)

    def tearDown(self):
        self.temporary.cleanup()

    def _build(self, *, rows=None, output_name="output"):
        if rows is not None:
            _write_jsonl(self.blocks_path, rows)
        output = self.root / output_name
        manifest = build_canonical_loo_prompts(
            analysis_blocks_file=self.blocks_path,
            section_roster_file=self.section_path,
            indicator_roster_file=self.roster_path,
            population_file=self.population_path,
            tokenizer=WhitespaceTokenizer(),
            tokenizer_artifact=TOKENIZER_ARTIFACT,
            output_dir=output,
        )
        return output, manifest

    def test_builds_complete_fixed_order_exact_delete_and_neutral_exports(self):
        output, manifest = self._build()
        context = "formal-recent-2-v1"
        baseline_path = (
            output / "exact_delete" / f"None_masked_{context}.jsonl"
        )
        baseline_rows = _read_jsonl(baseline_path)

        self.assertEqual(len(baseline_rows), 4)
        self.assertEqual(
            manifest["counts"],
            {
                "meeting_count": 2,
                "section_count": 2,
                "unit_count": 4,
                "indicator_count": 26,
                "analysis_block_count": 52,
                "full_prompt_count": 4,
                "delete_prompt_count": 104,
                "neutral_prompt_count": 104,
            },
        )
        full_prompt = baseline_rows[0]["prompt"]
        positions = [
            full_prompt.index(f"<<<LOO-INDICATOR-BLOCK:{indicator} |")
            for indicator in self.indicators
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(full_prompt.count("<<<LOO-INDICATOR-BLOCK:"), 26)
        self.assertEqual(
            baseline_rows[0]["sample_id"],
            "2024-01-31::Participants' Views",
        )

        indicator = self.indicators[0]
        delete_row = _read_jsonl(
            output
            / "exact_delete"
            / f"{indicator}_masked_{context}.jsonl"
        )[0]
        start, end, removed = single_contiguous_deletion(
            full_prompt,
            delete_row["prompt"],
        )
        self.assertTrue(has_line_block_boundaries(full_prompt, start, end))
        self.assertIn(indicator, removed)
        self.assertEqual(delete_row["arm"], "delete")
        self.assertEqual(delete_row["sample_id"], baseline_rows[0]["sample_id"])
        self.assertEqual(
            delete_row["prompt"].count("<<<LOO-INDICATOR-BLOCK:"),
            25,
        )

        neutral_row = _read_jsonl(
            output / "neutral" / f"{indicator}_masked_{context}.jsonl"
        )[0]
        self.assertEqual(neutral_row["arm"], "neutral")
        self.assertEqual(
            neutral_row["prompt_token_count_no_special_tokens"],
            baseline_rows[0]["prompt_token_count_no_special_tokens"],
        )
        self.assertNotIn(
            self.valid_rows[0]["generated"],
            neutral_row["prompt"],
        )
        self.assertIn(f"Indicator: {indicator.replace('-', ' ')}", neutral_row["prompt"])

        intervention = json.loads(
            (output / "intervention_manifest.json").read_text(encoding="utf-8")
        )
        neutral_manifest = json.loads(
            (output / "neutral_intervention_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(intervention["population_id"], "formal-recent-2-v1")
        self.assertEqual(len(intervention["interventions"]), 104)
        self.assertTrue(
            all(
                proof["line_block_boundaries_validated"]
                for proof in intervention["interventions"]
            )
        )
        self.assertEqual(len(neutral_manifest["replacements"]), 104)
        self.assertTrue(
            all(
                proof["token_length_matched"]
                and proof["prefix_suffix_unchanged"]
                and proof["label_and_delimiters_preserved"]
                for proof in neutral_manifest["replacements"]
            )
        )
        run_roster = json.loads(
            (output / "run_intervention_roster.json").read_text(encoding="utf-8")
        )
        self.assertEqual(run_roster["contexts"], [context])
        self.assertEqual(run_roster["indicators"], self.indicators)
        independently_validated = build_intervention_manifest(
            input_dir=(output / "exact_delete").resolve(),
            input_prompt_files=sorted((output / "exact_delete").glob("*.jsonl")),
            roster_file=(output / "run_intervention_roster.json").resolve(),
        )
        self.assertEqual(
            independently_validated["validation"],
            "exact_single_contiguous_prompt_deletion",
        )
        self.assertEqual(len(independently_validated["interventions"]), 104)

    def test_rejects_duplicate_or_incomplete_meeting_indicator_coverage(self):
        with self.subTest("duplicate"):
            rows = [*self.valid_rows, dict(self.valid_rows[0])]
            with self.assertRaisesRegex(ValueError, "Duplicate analysis block"):
                self._build(rows=rows, output_name="duplicate")

        with self.subTest("incomplete"):
            with self.assertRaisesRegex(ValueError, "complete frozen"):
                self._build(
                    rows=self.valid_rows[:-1],
                    output_name="incomplete",
                )

    def test_rejects_target_or_reference_text_fields(self):
        rows = [dict(row) for row in self.valid_rows]
        rows[0]["target"] = "historical Minutes text must not enter the prompt"
        with self.assertRaisesRegex(ValueError, "prohibited target/reference"):
            self._build(rows=rows, output_name="target-leak")

    def test_frozen_artifacts_are_idempotent_but_not_overwritable(self):
        output, first = self._build()
        _, second = self._build()
        self.assertEqual(first, second)

        manifest_path = output / "prompt_manifest.json"
        manifest_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self._build()


if __name__ == "__main__":
    unittest.main()
