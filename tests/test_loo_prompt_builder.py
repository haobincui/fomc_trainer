import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jobs.generation.loo_prompt_builder import build_canonical_loo_prompts
from jobs.generation.mask_generation import build_intervention_manifest
from open_r1.minutes_prompt import CANONICAL_MINUTES_PROMPT_SPEC
from open_r1.validator.intervention import (
    has_line_block_boundaries,
    single_contiguous_deletion,
)


class WhitespaceTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(text.split())))

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        if tokenize is not True:
            raise AssertionError("Tests require tokenized chat templates")
        token_count = sum(
            len(self.encode(f"{message['role']} {message['content']}"))
            for message in messages
        )
        if add_generation_prompt:
            token_count += 1
        return list(range(token_count))


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
        self.tokenizer = WhitespaceTokenizer()
        self.minutes_system_prompt = "Draft only the requested Minutes section."
        self.minutes_max_new_tokens = 8192
        self.minutes_max_model_len = 16384

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
                    f"PRIVATE REASONING for {meeting_date} and {indicator}; "
                    "this scratch work must never enter a Minutes prompt."
                    "\n</think>\n"
                    f"Projected final evidence for {meeting_date} and {indicator}."
                ),
                "minutes_analysis": (
                    f"Projected final evidence for {meeting_date} and "
                    f"{indicator} with several neutralizable words"
                ),
            }
            for meeting_date in self.dates
            for indicator in self.indicators
        ]
        _write_jsonl(self.blocks_path, self.valid_rows)

    def tearDown(self):
        self.temporary.cleanup()

    def _build(
        self,
        *,
        rows=None,
        output_name="output",
        minutes_max_new_tokens=None,
        minutes_max_model_len=None,
    ):
        if rows is not None:
            _write_jsonl(self.blocks_path, rows)
        output = self.root / output_name
        manifest = build_canonical_loo_prompts(
            analysis_blocks_file=self.blocks_path,
            section_roster_file=self.section_path,
            indicator_roster_file=self.roster_path,
            population_file=self.population_path,
            tokenizer=self.tokenizer,
            tokenizer_artifact=TOKENIZER_ARTIFACT,
            output_dir=output,
            analysis_text_field="minutes_analysis",
            minutes_max_new_tokens=(
                self.minutes_max_new_tokens
                if minutes_max_new_tokens is None
                else minutes_max_new_tokens
            ),
            minutes_max_model_len=(
                self.minutes_max_model_len
                if minutes_max_model_len is None
                else minutes_max_model_len
            ),
            minutes_system_prompt=self.minutes_system_prompt,
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
        self.assertIn(self.valid_rows[0]["minutes_analysis"], full_prompt)
        self.assertNotIn("PRIVATE REASONING", full_prompt)
        self.assertNotIn("</think>", full_prompt)
        expected_chat_tokens = len(
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.minutes_system_prompt},
                    {"role": "user", "content": full_prompt},
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        self.assertEqual(
            baseline_rows[0]["prompt_token_count_chat_template"],
            expected_chat_tokens,
        )
        self.assertEqual(
            baseline_rows[0]["context_headroom_tokens"],
            (
                self.minutes_max_model_len
                - self.minutes_max_new_tokens
                - expected_chat_tokens
            ),
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
            self.valid_rows[0]["minutes_analysis"],
            neutral_row["prompt"],
        )
        self.assertIn(f"Indicator: {indicator.replace('-', ' ')}", neutral_row["prompt"])

        self.assertEqual(manifest["schema_version"], "loo-prompt-manifest-v2")
        self.assertEqual(manifest["analysis_text_field"], "minutes_analysis")
        context_budget = manifest["context_budget"]
        self.assertEqual(
            context_budget["policy"],
            "exact-chat-template-no-truncation-v1",
        )
        self.assertEqual(
            context_budget["system_prompt_sha256"],
            hashlib.sha256(self.minutes_system_prompt.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            context_budget["token_count_policy"],
            "tokenizer.apply_chat_template(tokenize=True,"
            "add_generation_prompt=True)",
        )
        self.assertEqual(
            context_budget["max_new_tokens"],
            self.minutes_max_new_tokens,
        )
        self.assertEqual(
            context_budget["max_model_len"],
            self.minutes_max_model_len,
        )
        self.assertEqual(context_budget["audited_prompt_row_count"], 216)
        self.assertTrue(context_budget["all_rows_fit"])
        self.assertEqual(context_budget["input_truncation"], "forbidden")

        audited_rows = [
            row
            for artifact in manifest["artifacts"]
            for row in _read_jsonl(output / artifact["relative_path"])
        ]
        self.assertEqual(len(audited_rows), 216)
        self.assertEqual(
            context_budget["maximum_chat_prompt_tokens"],
            max(row["prompt_token_count_chat_template"] for row in audited_rows),
        )
        self.assertEqual(
            context_budget["minimum_context_headroom_tokens"],
            min(row["context_headroom_tokens"] for row in audited_rows),
        )
        self.assertTrue(
            all(
                row["prompt_token_count_chat_template"]
                + self.minutes_max_new_tokens
                + row["context_headroom_tokens"]
                == self.minutes_max_model_len
                for row in audited_rows
            )
        )

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

    def test_rejects_chat_template_context_budget_overflow(self):
        with self.assertRaisesRegex(
            ValueError,
            (
                "Canonical Minutes prompt exceeds the frozen context budget: "
                ".*chat_prompt_tokens=.*overflow="
            ),
        ):
            self._build(
                output_name="context-overflow",
                minutes_max_new_tokens=10,
                minutes_max_model_len=11,
            )

    def test_scoped_config_binds_exact_prompt_to_manifest_and_rows(self):
        real_roster_id = "chapter2-historical-26-indicators-after-2009-v1"
        _write_json(
            self.roster_path,
            {
                "schema_version": "loo-intervention-roster-v1",
                "roster_id": real_roster_id,
                "baseline_indicator": "None",
                "contexts": ["source-context"],
                "indicators": self.indicators,
                "indicator_markers": {
                    indicator: [indicator.replace("-", " ")]
                    for indicator in self.indicators
                },
            },
        )
        experiment_config = self.root / "experiment.json"
        _write_json(
            experiment_config,
            {
                "schema_version": "loo-scoped-experiment-v1",
                "experiment_id": "synthetic-scope-v1",
                "baseline_indicator": "None",
                "full_context_roster_id": real_roster_id,
                "full_context_indicators": self.indicators,
                "intervention_indicators": self.indicators[:6],
                "populations": {
                    "formal-recent-2-v1": {
                        "phase": "formal",
                        "split_label": "test",
                        "meeting_dates": self.dates,
                    }
                },
                "section_names": self.sections,
                "minutes_system_prompt": CANONICAL_MINUTES_PROMPT_SPEC.as_dict(),
                "decoding": {},
                "release_policy": {},
                "claim_boundary": "synthetic test",
            },
        )
        output = self.root / "scoped-output"
        manifest = build_canonical_loo_prompts(
            analysis_blocks_file=self.blocks_path,
            section_roster_file=self.section_path,
            indicator_roster_file=self.roster_path,
            population_file=self.population_path,
            tokenizer=self.tokenizer,
            tokenizer_artifact=TOKENIZER_ARTIFACT,
            output_dir=output,
            analysis_text_field="minutes_analysis",
            experiment_config_file=experiment_config,
        )

        self.assertEqual(
            manifest["minutes_system_prompt"],
            CANONICAL_MINUTES_PROMPT_SPEC.as_dict(),
        )
        self.assertEqual(
            manifest["experiment_config"]["sha256"],
            hashlib.sha256(experiment_config.read_bytes()).hexdigest(),
        )
        first_row = _read_jsonl(
            output
            / "exact_delete"
            / "None_masked_formal-recent-2-v1.jsonl"
        )[0]
        self.assertEqual(
            first_row["generation_system_prompt_sha256"],
            CANONICAL_MINUTES_PROMPT_SPEC.sha256,
        )
        self.assertEqual(
            first_row["generation_requested_max_output_tokens"],
            4096,
        )

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
