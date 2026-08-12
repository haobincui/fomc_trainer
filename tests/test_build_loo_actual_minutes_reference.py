import json
import tempfile
import unittest
from pathlib import Path

from jobs.eval.build_loo_actual_minutes_reference import (
    build_actual_minutes_reference,
)
from open_r1.provenance import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPE = REPO_ROOT / "configs/main/loo_experiment_legacy6.json"
POPULATION = REPO_ROOT / "configs/main/loo_population_pilot_eval_13.json"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestBuildActualMinutesReference(unittest.TestCase):
    def _source_rows(self) -> list[dict]:
        scope = json.loads(SCOPE.read_text(encoding="utf-8"))
        rows = []
        for meeting_date in scope["populations"]["pilot_eval_13"]["meeting_dates"]:
            for section_name in scope["section_names"]:
                source_section = section_name
                if (
                    meeting_date == "2021-12-15"
                    and section_name.startswith("Participants' Views")
                ):
                    source_section = (
                        "Participants' Views on Current Economic Conditions and "
                        "the Economic Outlook"
                    )
                rows.append(
                    {
                        "meeting_date": meeting_date,
                        "section_name": source_section,
                        "reference": f"Actual Minutes for {meeting_date} / {section_name}",
                    }
                )
        return rows

    def test_builds_deterministic_scoped_39_row_view_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jsonl"
            output = root / "reference.jsonl"
            manifest_path = root / "reference_manifest.json"
            _write_jsonl(source, self._source_rows())

            manifest = build_actual_minutes_reference(
                source_file=source,
                population_manifest_file=POPULATION,
                scope_manifest_file=SCOPE,
                population_id="pilot_eval_13",
                output_file=output,
                manifest_file=manifest_path,
                expected_source_sha256=sha256_file(source),
            )

            output_rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(output_rows), 39)
            self.assertEqual(manifest["row_count"], 39)
            self.assertEqual(manifest["output"]["sha256"], sha256_file(output))
            self.assertEqual(
                output_rows[0]["section_name"],
                "Participants' Views on Current Conditions and the Economic Outlook",
            )
            self.assertIn(
                "Participants' Views on Current Economic Conditions and the Economic Outlook",
                manifest["section_name_aliases"],
            )

    def test_rejects_incomplete_reference_matrix(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jsonl"
            rows = self._source_rows()
            _write_jsonl(source, rows[:-1])
            with self.assertRaisesRegex(ValueError, "13x3 matrix"):
                build_actual_minutes_reference(
                    source_file=source,
                    population_manifest_file=POPULATION,
                    scope_manifest_file=SCOPE,
                    population_id="pilot_eval_13",
                    output_file=root / "reference.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
