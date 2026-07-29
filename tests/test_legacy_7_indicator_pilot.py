import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from jobs.eval.eval_legacy_7_indicator_pilot import (
    EXCEL_CELL_CHARACTER_LIMIT,
    TruncatedLegacyResponseError,
    _summarise,
    load_legacy_pilot_rows,
    parse_legacy_assistant_content,
    run_legacy_rescore,
    score_records_from_embeddings,
)
from open_r1.provenance import sha256_text


def _assistant_repr(content: str) -> str:
    return repr({"role": "assistant", "content": content})


class TestLegacyResponseParsing(unittest.TestCase):
    def test_parses_python_repr_without_changing_content(self):
        content = "Exact text with apostrophe's and a newline.\n"

        parsed = parse_legacy_assistant_content(
            _assistant_repr(content),
            source="fixture",
        )

        self.assertEqual(parsed, content)

    def test_excel_limit_parse_failure_is_an_explicit_truncation(self):
        truncated = "{" + ("x" * (EXCEL_CELL_CHARACTER_LIMIT - 1))

        with self.assertRaises(TruncatedLegacyResponseError):
            parse_legacy_assistant_content(truncated, source="fixture")

    def test_other_parse_failures_are_fatal(self):
        with self.assertRaisesRegex(ValueError, "valid Python literal"):
            parse_legacy_assistant_content("{broken", source="fixture")


class TestLegacyPilotExtraction(unittest.TestCase):
    def _build_fixture(self, root: Path) -> None:
        training_rows = [
            {
                "instruction": "instruction one",
                "input": "",
                "output": "historical actual target one",
            },
            {
                "instruction": "instruction two",
                "input": "",
                "output": "historical actual target two",
            },
        ]
        tagged_rows = [
            {
                "instruction": "instruction one",
                "input": "",
                "output": "cleaned metadata text one",
                "section_tag": "Staff-Review-of-the-Economic-Situation",
            },
            {
                "instruction": "instruction two",
                "input": "",
                "output": "cleaned metadata text two",
                "section_tag": "Staff-Review-of-the-Financial-Situation",
            },
        ]
        training_path = root / "data/training/input_qa.json"
        tagged_path = root / "output/archive/qa_json/input_qa_tagged.json"
        training_path.parent.mkdir(parents=True)
        tagged_path.parent.mkdir(parents=True)
        training_path.write_text(json.dumps(training_rows), encoding="utf-8")
        tagged_path.write_text(json.dumps(tagged_rows), encoding="utf-8")

        section_specs = (
            (
                "20240131",
                "Staff-Review-of-the-Economic-Situation",
                "cleaned metadata text one",
            ),
            (
                "20240320",
                "Staff-Review-of-the-Financial-Situation",
                "cleaned metadata text two",
            ),
        )
        for date, section, text in section_specs:
            path = (
                root
                / "data/processed/sections"
                / f"fomcminutes{date}_cleaned_sections"
                / f"{section}.txt"
            )
            path.parent.mkdir(parents=True)
            path.write_text(text, encoding="utf-8")

        full_responses = [_assistant_repr("full one"), _assistant_repr("full two")]
        full_path = (
            root
            / "output/archive/generated/section_detail_responses/ft_20250330"
            / "generated_1.xlsx"
        )
        full_path.parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "index": [0, 1],
                "prompts": ["prompt one", "prompt two"],
                "targets": [
                    "historical actual target one",
                    "historical actual target two",
                ],
                "generateds": full_responses,
            }
        ).to_excel(full_path, index=False)

        truncated = "{" + ("x" * (EXCEL_CELL_CHARACTER_LIMIT - 1))
        mask_path = (
            root / "output/archive/masked/ft_20250330" / "mask_gdp_generated_1.xlsx"
        )
        mask_path.parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "index": [0, 1],
                "unmask_prompt": ["prompt one", "prompt two"],
                "unmask_response": [
                    response.split("'content':")[-1] for response in full_responses
                ],
                "mask_prompt": ["masked prompt one", "masked prompt two"],
                "mask_response": [_assistant_repr("masked one"), truncated],
            }
        ).to_excel(mask_path, index=False)

    def test_uses_training_target_and_excludes_truncated_mask(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._build_fixture(root)

            rows, exclusions, audit = load_legacy_pilot_rows(
                root,
                indicators=("gdp",),
                batches=(1,),
                expected_rows_per_workbook=2,
                strict_known_hashes=False,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(exclusions), 1)
        self.assertEqual(rows[0]["target"], "historical actual target one")
        self.assertNotEqual(rows[0]["target"], "cleaned metadata text one")
        self.assertEqual(rows[0]["full_output"], "full one")
        self.assertEqual(rows[0]["masked_output"], "masked one")
        self.assertEqual(rows[0]["meeting_date"], "2024-01-31")
        self.assertEqual(
            exclusions[0]["exclusion_reason"],
            "excel_cell_truncated_masked_output",
        )
        self.assertEqual(audit["counts"]["candidate_interventions"], 2)
        self.assertEqual(audit["counts"]["retained_interventions"], 1)
        metadata_binding = audit["metadata_source"]["derived_qa_index_mapping"]
        self.assertEqual(metadata_binding["row_count"], 2)
        self.assertEqual(metadata_binding["unique_section_files"], 2)
        self.assertEqual(len(metadata_binding["sha256"]), 64)


class TestLegacyPilotScoring(unittest.TestCase):
    def test_signed_delta_uses_the_same_actual_target(self):
        target = "actual"
        full = "full"
        masked = "masked"
        record = {
            "indicator": "gdp",
            "meeting_date": "2024-01-31",
            "section_name": "section",
            "qa_index": 1,
            "legacy_batch": 1,
            "target": target,
            "target_sha256": sha256_text(target),
            "full_output": full,
            "full_output_sha256": sha256_text(full),
            "masked_output": masked,
            "masked_output_sha256": sha256_text(masked),
        }
        embeddings = {
            sha256_text(target): np.asarray([1.0, 0.0], dtype=np.float32),
            sha256_text(full): np.asarray([0.8, 0.6], dtype=np.float32),
            sha256_text(masked): np.asarray([0.6, 0.8], dtype=np.float32),
        }

        scored = score_records_from_embeddings(
            [record],
            embeddings,
            embedding_model_path="model",
            embedding_model_sha256="a" * 64,
        )[0]

        self.assertAlmostEqual(scored["similarity_full"], 0.8)
        self.assertAlmostEqual(scored["similarity_masked"], 0.6)
        self.assertAlmostEqual(scored["delta"], 0.2)
        self.assertAlmostEqual(scored["distance_masked"] - scored["distance_full"], 0.2)
        self.assertNotIn("target", scored)
        self.assertNotIn("full_output", scored)
        self.assertNotIn("masked_output", scored)

    def test_summary_reports_row_and_meeting_balanced_means(self):
        rows = []
        for meeting, qa_index, delta in (
            ("2024-01-31", 1, 0.1),
            ("2024-01-31", 2, 0.3),
            ("2024-03-20", 3, 0.8),
        ):
            rows.append(
                {
                    "indicator": "gdp",
                    "meeting_date": meeting,
                    "section_name": "section",
                    "qa_index": qa_index,
                    "legacy_batch": 1,
                    "similarity_full": 0.9,
                    "similarity_masked": 0.9 - delta,
                    "delta": delta,
                    "self_distance": 0.2,
                }
            )

        summary = _summarise(rows, ("indicator",))[0]

        self.assertAlmostEqual(summary["mean_delta_row_weighted"], 0.4)
        self.assertAlmostEqual(summary["mean_delta_meeting_balanced"], 0.5)
        self.assertEqual(summary["n_meetings"], 2)


class TestLegacyPilotRunStatus(unittest.TestCase):
    def test_full_run_marks_running_before_invalidating_old_results(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            with (
                patch.object(
                    Path,
                    "unlink",
                    side_effect=RuntimeError("simulated invalidation crash"),
                ),
                self.assertRaisesRegex(RuntimeError, "simulated invalidation crash"),
            ):
                run_legacy_rescore(
                    source_root=output_dir / "unused",
                    output_dir=output_dir,
                    embedding_model_path=None,
                )

            status = json.loads(
                (output_dir / "run_status.json").read_text(encoding="utf-8")
            )

        self.assertEqual(status["mode"], "full_rescore")
        self.assertEqual(status["status"], "running")

    def test_validate_only_does_not_replace_prior_full_run_status(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            prior_status = {
                "mode": "full_rescore",
                "status": "complete",
                "execution_id": "prior",
            }
            (output_dir / "run_status.json").write_text(
                json.dumps(prior_status),
                encoding="utf-8",
            )
            (output_dir / "results.md").write_text(
                "prior complete results",
                encoding="utf-8",
            )
            prior_exclusions = "prior full-run exclusions\n"
            (output_dir / "exclusions.jsonl").write_text(
                prior_exclusions,
                encoding="utf-8",
            )
            with patch(
                "jobs.eval.eval_legacy_7_indicator_pilot.load_legacy_pilot_rows",
                return_value=([], [], {"fixture": True}),
            ):
                run_legacy_rescore(
                    source_root=output_dir / "unused",
                    output_dir=output_dir,
                    embedding_model_path=None,
                    validate_only=True,
                )

            observed_prior_status = json.loads(
                (output_dir / "run_status.json").read_text(encoding="utf-8")
            )
            validation_status = json.loads(
                (output_dir / "input_validation_status.json").read_text(
                    encoding="utf-8"
                )
            )
            prior_results = (output_dir / "results.md").read_text(encoding="utf-8")
            observed_exclusions = (output_dir / "exclusions.jsonl").read_text(
                encoding="utf-8"
            )

        self.assertEqual(observed_prior_status, prior_status)
        self.assertEqual(prior_results, "prior complete results")
        self.assertEqual(observed_exclusions, prior_exclusions)
        self.assertEqual(validation_status["mode"], "validate_only")
        self.assertEqual(validation_status["status"], "validated_inputs")


if __name__ == "__main__":
    unittest.main()
