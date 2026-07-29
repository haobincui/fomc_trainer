import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from jobs.generation.canonical_indicator_analysis import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    IndicatorAnalysisDecoding,
    build_indicator_analysis_prompt,
    run_indicator_analysis_generation,
    validate_and_prepare_rows,
    validate_ledger_provenance,
)
from open_r1.provenance import sha256_file, sha256_text
from open_r1.validator.loo_generation_spec import seal_manifest


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _row(
    meeting_id: str,
    indicator: str,
    *,
    information_as_of_date: str | None = None,
    observation_date: str = "2023-10-01",
) -> dict:
    as_of = information_as_of_date or (
        date.fromisoformat(meeting_id) - timedelta(days=1)
    ).isoformat()
    payload = {
        "series": [
            {
                "source_key": f"example-{indicator}",
                "series_id": "EXAMPLE",
                "title": indicator,
                "frequency": "monthly",
                "units": "percent",
                "seasonal_adjustment": "none",
                "transformation": "level",
                "availability_as_of_date": as_of,
                "requested_vintage_date": as_of,
                "observations": [
                    {"date": observation_date, "value": "3.2"}
                ],
            }
        ]
    }
    return {
        "schema_version": "canonical-loo-indicator-input-v2",
        "meeting_id": meeting_id,
        "sample_id": f"{meeting_id}::{indicator}",
        "meeting_timestamp": f"{meeting_id}T19:00:00Z",
        "meeting_date": meeting_id,
        "indicator": indicator,
        "source_id": f"example:{indicator}",
        "source_sha256": sha256_text(f"raw-source:{indicator}"),
        "source_timestamp": "2026-07-28T00:00:00Z",
        "information_as_of_date": as_of,
        "requested_vintage_date": as_of,
        "availability_as_of_date": as_of,
        "availability_evidence_type": "alfred_vintage_snapshot",
        "source_interface": "alfred-graph-csv-v1",
        "observation_date": observation_date,
        "source_payload": payload,
    }


class RecordingGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], str, dict]] = []

    def __call__(self, rows, model_path, **kwargs):
        self.calls.append((rows, model_path, kwargs))
        return [
            {
                **row,
                "generated": f"Analysis for {row['sample_id']}",
            }
            for row in rows
        ]


class TestCanonicalIndicatorAnalysisValidation(unittest.TestCase):
    def test_rejects_information_date_after_d_minus_one(self):
        rows = [
            _row(
                "2024-01-31",
                "GDP-Growth",
                information_as_of_date="2024-01-31",
            )
        ]

        with self.assertRaisesRegex(ValueError, "meeting_date - 1 day"):
            validate_and_prepare_rows(rows, ["GDP-Growth"])

    def test_rejects_observation_after_d_minus_one(self):
        row = _row("2024-01-31", "GDP-Growth")
        row["observation_date"] = "2024-01-31"
        row["source_payload"]["series"][0]["observations"][0][
            "date"
        ] = "2024-01-31"

        with self.assertRaisesRegex(ValueError, "observation_date is after D-1"):
            validate_and_prepare_rows([row], ["GDP-Growth"])

    def test_rejects_meeting_day_vintage_even_for_early_release(self):
        row = _row("2024-01-31", "GDP-Growth")
        row["requested_vintage_date"] = "2024-01-31"

        with self.assertRaisesRegex(ValueError, "requested_vintage_date"):
            validate_and_prepare_rows([row], ["GDP-Growth"])

    def test_rejects_current_revision_schema(self):
        row = _row("2024-01-31", "GDP-Growth")
        row["schema_version"] = "legacy-current-vintage"

        with self.assertRaisesRegex(ValueError, "canonical-loo-indicator-input-v2"):
            validate_and_prepare_rows([row], ["GDP-Growth"])

    def test_rejects_incomplete_meeting_roster(self):
        rows = [_row("2024-01-31", "GDP-Growth")]

        with self.assertRaisesRegex(ValueError, "Incomplete indicator roster"):
            validate_and_prepare_rows(
                rows,
                ["GDP-Growth", "Unemployment-Rate"],
            )

    def test_rejects_unstable_sample_id(self):
        row = _row("2024-01-31", "GDP-Growth")
        row["sample_id"] = "row-7"

        with self.assertRaisesRegex(ValueError, "stable sample_id"):
            validate_and_prepare_rows([row], ["GDP-Growth"])

    def test_rejects_meeting_id_that_is_not_the_meeting_date(self):
        row = _row("2024-01-31", "GDP-Growth")
        row["meeting_id"] = "meeting-2024-01-31"
        row["sample_id"] = "meeting-2024-01-31::GDP-Growth"

        with self.assertRaisesRegex(ValueError, "meeting_id must equal"):
            validate_and_prepare_rows([row], ["GDP-Growth"])

    def test_prompt_is_deterministic_for_equivalent_payload_key_order(self):
        first = _row("2024-01-31", "GDP-Growth")
        second = _row("2024-01-31", "GDP-Growth")
        first_series = first["source_payload"]["series"]
        second_series = second["source_payload"]["series"]
        first["source_payload"] = {
            "series": first_series,
            "sampling_policy": "native-level-v1",
        }
        second["source_payload"] = {
            "sampling_policy": "native-level-v1",
            "series": second_series,
        }

        first_prepared = validate_and_prepare_rows([first], ["GDP-Growth"])[0]
        second_prepared = validate_and_prepare_rows([second], ["GDP-Growth"])[0]

        self.assertEqual(
            build_indicator_analysis_prompt(first_prepared),
            build_indicator_analysis_prompt(second_prepared),
        )
        self.assertEqual(
            first_prepared["prompt_sha256"],
            second_prepared["prompt_sha256"],
        )

    def test_prompt_excludes_post_meeting_retrieval_timestamp(self):
        prepared = validate_and_prepare_rows(
            [_row("2024-01-31", "GDP-Growth")],
            ["GDP-Growth"],
        )[0]

        self.assertEqual(prepared["source_timestamp"], "2026-07-28T00:00:00Z")
        self.assertNotIn("release_timestamp", prepared)
        self.assertNotIn("2026-07-28", prepared["prompt"])
        self.assertIn("Information available through: 2024-01-30", prepared["prompt"])


class TestCanonicalIndicatorAnalysisRun(unittest.TestCase):
    def test_generates_complete_panel_with_fingerprinted_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = root / "model"
            tokenizer = root / "tokenizer"
            model.mkdir()
            tokenizer.mkdir()
            (model / "weights.bin").write_bytes(b"frozen weights")
            (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
            roster_path = root / "roster.json"
            input_path = root / "input.jsonl"
            output_dir = root / "run"
            roster = ["GDP-Growth", "Unemployment-Rate"]
            _write_json(roster_path, {"indicators": roster})
            rows = [
                _row(meeting, indicator)
                for meeting in ("2024-01-31", "2024-03-20")
                for indicator in reversed(roster)
            ]
            _write_jsonl(input_path, rows)
            model_hash_before = sha256_file(model / "weights.bin")
            tokenizer_hash_before = sha256_file(tokenizer / "tokenizer.json")
            generator = RecordingGenerator()

            manifest = run_indicator_analysis_generation(
                input_jsonl=input_path,
                roster_json=roster_path,
                model_path=model,
                tokenizer_path=tokenizer,
                output_dir=output_dir,
                generation_fn=generator,
            )

            generated_rows = _read_jsonl(output_dir / "indicator_analysis.jsonl")
            disk_manifest = json.loads(
                (output_dir / "analysis_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(generated_rows), 4)
            self.assertEqual(manifest, disk_manifest)
            self.assertTrue(manifest["generation_only"])
            self.assertFalse(manifest["training_performed"])
            self.assertEqual(manifest["inventory"]["meeting_count"], 2)
            self.assertEqual(manifest["inventory"]["indicator_count"], 2)
            self.assertEqual(manifest["inventory"]["row_count"], 4)
            self.assertEqual(
                manifest["output"]["sha256"],
                sha256_file(output_dir / "indicator_analysis.jsonl"),
            )
            self.assertEqual(
                [row["indicator"] for row in generated_rows],
                roster + roster,
            )
            self.assertTrue(
                all(row["run_id"] == manifest["run_id"] for row in generated_rows)
            )
            self.assertEqual(sha256_file(model / "weights.bin"), model_hash_before)
            self.assertEqual(
                sha256_file(tokenizer / "tokenizer.json"),
                tokenizer_hash_before,
            )
            self.assertEqual(len(generator.calls), 1)
            _, resolved_model_path, kwargs = generator.calls[0]
            self.assertEqual(resolved_model_path, str(model.resolve()))
            self.assertEqual(kwargs["temperature"], DEFAULT_TEMPERATURE)
            self.assertEqual(kwargs["top_p"], DEFAULT_TOP_P)
            self.assertEqual(
                kwargs["max_new_tokens"],
                DEFAULT_MAX_NEW_TOKENS,
            )

    def test_refuses_to_overwrite_immutable_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = root / "model"
            tokenizer = root / "tokenizer"
            model.mkdir()
            tokenizer.mkdir()
            (model / "weights.bin").write_bytes(b"weights")
            (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
            roster_path = root / "roster.json"
            input_path = root / "input.jsonl"
            output_dir = root / "run"
            output_dir.mkdir()
            (output_dir / "indicator_analysis.jsonl").write_text(
                "existing",
                encoding="utf-8",
            )
            _write_json(roster_path, {"indicators": ["GDP-Growth"]})
            _write_jsonl(input_path, [_row("2024-01-31", "GDP-Growth")])

            with self.assertRaisesRegex(FileExistsError, "immutable"):
                run_indicator_analysis_generation(
                    input_jsonl=input_path,
                    roster_json=roster_path,
                    model_path=model,
                    tokenizer_path=tokenizer,
                    output_dir=output_dir,
                    generation_fn=RecordingGenerator(),
                )

    def test_rejects_incomplete_generator_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = root / "model"
            tokenizer = root / "tokenizer"
            model.mkdir()
            tokenizer.mkdir()
            (model / "weights.bin").write_bytes(b"weights")
            (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
            roster_path = root / "roster.json"
            input_path = root / "input.jsonl"
            _write_json(roster_path, {"indicators": ["GDP-Growth"]})
            _write_jsonl(input_path, [_row("2024-01-31", "GDP-Growth")])

            def empty_generator(rows, model_path, **kwargs):
                return []

            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                run_indicator_analysis_generation(
                    input_jsonl=input_path,
                    roster_json=roster_path,
                    model_path=model,
                    tokenizer_path=tokenizer,
                    output_dir=root / "run",
                    generation_fn=empty_generator,
                )

    def test_rejects_meetings_outside_frozen_population_before_generation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = root / "model"
            tokenizer = root / "tokenizer"
            model.mkdir()
            tokenizer.mkdir()
            (model / "weights.bin").write_bytes(b"weights")
            (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
            roster_path = root / "roster.json"
            input_path = root / "input.jsonl"
            population_path = root / "population.json"
            _write_json(roster_path, {"indicators": ["GDP-Growth"]})
            _write_jsonl(input_path, [_row("2024-01-31", "GDP-Growth")])
            _write_json(
                population_path,
                {
                    "population_id": "formal",
                    "meeting_dates": ["2024-03-20"],
                },
            )
            generator = RecordingGenerator()

            with self.assertRaisesRegex(
                ValueError,
                "do not match the frozen population",
            ):
                run_indicator_analysis_generation(
                    input_jsonl=input_path,
                    roster_json=roster_path,
                    model_path=model,
                    tokenizer_path=tokenizer,
                    population_json=population_path,
                    output_dir=root / "run",
                    generation_fn=generator,
                )

            self.assertEqual(generator.calls, [])

    def test_non_deterministic_decoding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "temperature=0"):
            IndicatorAnalysisDecoding(temperature=0.5).validate()


class TestCanonicalIndicatorLedgerProvenance(unittest.TestCase):
    def _write_chain(self, root: Path) -> dict[str, Path]:
        registry = root / "inputs" / "registry.json"
        roster = root / "inputs" / "roster.json"
        population = root / "inputs" / "population.json"
        snapshot = root / "snapshots" / "snapshot_manifest.json"
        ledger_dir = root / "ledger"
        indicator_input = ledger_dir / "indicator_inputs.jsonl"
        ledger = ledger_dir / "ledger_manifest.json"

        _write_json(registry, {"schema_version": "registry-fixture-v1"})
        _write_json(roster, {"indicators": ["GDP-Growth"]})
        _write_json(
            population,
            {
                "population_id": "fixture",
                "meeting_dates": ["2024-01-31"],
            },
        )
        _write_jsonl(
            indicator_input,
            [_row("2024-01-31", "GDP-Growth")],
        )
        _write_json(
            snapshot,
            seal_manifest(
                {
                    "schema_version": "loo-source-snapshot-manifest-v1",
                    "status": "complete",
                    "registry": {"sha256": sha256_file(registry)},
                }
            ),
        )
        _write_json(
            ledger,
            seal_manifest(
                {
                    "schema_version": "loo-indicator-ledger-manifest-v1",
                    "status": "complete",
                    "inputs": {
                        "registry": {"sha256": sha256_file(registry)},
                        "snapshot_manifest": {
                            "sha256": sha256_file(snapshot)
                        },
                        "roster": {"sha256": sha256_file(roster)},
                        "population": {
                            "sha256": sha256_file(population)
                        },
                    },
                    "outputs": {
                        "indicator_inputs": {
                            "path": "indicator_inputs.jsonl",
                            "sha256": sha256_file(indicator_input),
                        }
                    },
                }
            ),
        )
        return {
            "registry": registry.resolve(),
            "roster": roster.resolve(),
            "population": population.resolve(),
            "snapshot": snapshot.resolve(),
            "ledger": ledger.resolve(),
            "input": indicator_input.resolve(),
        }

    def test_accepts_a_sealed_hash_bound_d1_ledger_chain(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self._write_chain(Path(temporary_directory))

            binding = validate_ledger_provenance(
                input_path=paths["input"],
                ledger_manifest_path=paths["ledger"],
                snapshot_manifest_path=paths["snapshot"],
                source_registry_path=paths["registry"],
                roster_path=paths["roster"],
                population_path=paths["population"],
            )

            self.assertEqual(
                binding["indicator_input_sha256"],
                sha256_file(paths["input"]),
            )
            self.assertEqual(
                binding["source_registry_sha256"],
                sha256_file(paths["registry"]),
            )

    def test_rejects_a_snapshot_changed_after_ledger_sealing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self._write_chain(Path(temporary_directory))
            paths["snapshot"].write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "[Mm]anifest|hash mismatch",
            ):
                validate_ledger_provenance(
                    input_path=paths["input"],
                    ledger_manifest_path=paths["ledger"],
                    snapshot_manifest_path=paths["snapshot"],
                    source_registry_path=paths["registry"],
                    roster_path=paths["roster"],
                    population_path=paths["population"],
                )

    def test_rejects_a_roster_changed_after_ledger_sealing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self._write_chain(Path(temporary_directory))
            _write_json(
                paths["roster"],
                {"indicators": ["Unemployment-Rate"]},
            )

            with self.assertRaisesRegex(ValueError, "roster hash mismatch"):
                validate_ledger_provenance(
                    input_path=paths["input"],
                    ledger_manifest_path=paths["ledger"],
                    snapshot_manifest_path=paths["snapshot"],
                    source_registry_path=paths["registry"],
                    roster_path=paths["roster"],
                    population_path=paths["population"],
                )


if __name__ == "__main__":
    unittest.main()
