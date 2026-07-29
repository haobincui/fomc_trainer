import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from open_r1.utils.main_pipeline import (
    build_meeting_random_split,
    deduplicate_decision_rows,
    extract_meeting_date,
    split_analysis_train_for_grpo,
)
from process_fomc_report.generate_prompt_and_response.run_generate_prompt_pipeline import (
    build_parser,
    _iter_label_inputs,
    parse_binary_bool,
)


class TestMainPipelineUtils(unittest.TestCase):
    def test_parse_binary_bool_accepts_only_zero_or_one(self):
        self.assertFalse(parse_binary_bool("0"))
        self.assertTrue(parse_binary_bool("1"))

        with self.assertRaises(Exception):
            parse_binary_bool("true")
        with self.assertRaises(Exception):
            parse_binary_bool("2")

    def test_build_meeting_random_split_is_deterministic(self):
        start = date(2024, 1, 1)
        meetings = [(start + timedelta(days=index)).isoformat() for index in range(128)]
        split_one = build_meeting_random_split(meetings, seed=42)
        split_two = build_meeting_random_split(meetings, seed=42)

        self.assertEqual(split_one, split_two)
        self.assertEqual({key: len(value) for key, value in split_one.items()}, {"train": 102, "eval": 13, "test": 13})
        self.assertEqual(set(split_one["train"]) & set(split_one["eval"]), set())
        self.assertEqual(set(split_one["train"]) & set(split_one["test"]), set())
        self.assertEqual(set(split_one["eval"]) & set(split_one["test"]), set())

    def test_deduplicate_decision_rows_prefers_non_empty_response_then_lowest_index(self):
        rows = [
            {
                "index": 7,
                "prompt": "meeting on **2024-01-31**",
                "response": "",
                "rate_change": "No change",
                "current_rate": 5.25,
            },
            {
                "index": 3,
                "prompt": "meeting on **2024-01-31**",
                "response": "filled",
                "rate_change": "No change",
                "current_rate": 5.25,
            },
            {
                "index": 4,
                "prompt": "meeting on **2024-01-31**",
                "response": "filled later",
                "rate_change": "No change",
                "current_rate": 5.25,
            },
        ]

        deduped, audit = deduplicate_decision_rows(rows)

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["index"], 3)
        self.assertEqual(deduped[0]["deduplicated_from"], 3)
        self.assertEqual(audit["duplicate_meetings"], 1)

    def test_split_analysis_train_for_grpo_uses_primary_and_replay_subsets(self):
        rows = [
            {"index": index, "meeting_date": f"2024-01-{index + 1:02d}", "prompt": f"p{index}", "prompt_hash": f"h{index}"}
            for index in range(10)
        ]
        selected, manifest = split_analysis_train_for_grpo(rows, seed=42, primary_ratio=0.2, replay_ratio=0.1)

        roles = [row["analysis_grpo_role"] for row in selected]
        self.assertEqual(manifest["primary_row_count"], 2)
        self.assertEqual(manifest["replay_row_count"], 1)
        self.assertCountEqual(roles, ["grpo_primary", "grpo_primary", "sft_replay"])

    def test_extract_meeting_date_from_prompt_variants(self):
        prompt_variants = [
            "You are preparing for the meeting on **2024-03-20**.",
            "Upcoming meeting scheduled for **2024-03-20**.",
            "Decision prompt for the **2024-03-20** meeting.",
        ]

        for prompt in prompt_variants:
            self.assertEqual(extract_meeting_date(prompt), "2024-03-20")

    def test_iter_label_inputs_falls_back_to_nested_scope_directory(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested_root = root / "reformatted_html" / "reformted_html_files" / "after_2009"
            nested_root.mkdir(parents=True, exist_ok=True)
            expected = nested_root / "fomcminutes20090128.xlsx"
            expected.write_text("placeholder", encoding="utf-8")

            config = {
                "labeling": {"input_glob": ""},
                "pipeline": {"raw_html_root": str(root / "reformatted_html")},
            }

            resolved = _iter_label_inputs(config, "after_2009")

            self.assertEqual(resolved, [expected.resolve()])

    def test_force_teacher_refresh_parser_requires_explicit_zero_or_one(self):
        parser = build_parser()

        parsed_false = parser.parse_args(["--stage", "analysis_sft_teacher_responses", "--force-teacher-refresh=0"])
        parsed_true = parser.parse_args(["--stage", "analysis_sft_teacher_responses", "--force-teacher-refresh=1"])
        self.assertFalse(parsed_false.force_teacher_refresh)
        self.assertTrue(parsed_true.force_teacher_refresh)

        with self.assertRaises(SystemExit):
            parser.parse_args(["--stage", "analysis_sft_teacher_responses", "--force-teacher-refresh"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--stage", "analysis_sft_teacher_responses", "--force-teacher-refresh=true"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--stage", "analysis_sft_teacher_responses", "--force-teacher-refresh=2"])


if __name__ == "__main__":
    unittest.main()
