import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jobs.generation.mask_generation import run_mask_generation
from open_r1.provenance import (
    fingerprint_artifact_path,
    sha256_file,
    sha256_text,
)
from open_r1.validator.neutral_intervention import (
    validate_neutral_intervention_manifest,
)


class WhitespaceTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(text.split())))


def _json_text(payload) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _write_json(path: Path, payload) -> None:
    path.write_text(_json_text(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class NeutralFixture:
    context = "formal-test"
    baseline = "None"
    indicators = ["GDP-Growth", "Treasury-Yields"]
    meeting_date = "2024-01-31"
    section = "Participants' Views"
    sample_id = f"{meeting_date}::{section}"

    def __init__(self, root: Path):
        self.root = root
        self.prompt_dir = root / "neutral"
        self.prompt_dir.mkdir()
        self.roster_path = root / "run_roster.json"
        self.roster = {
            "schema_version": "loo-intervention-roster-v1",
            "roster_id": "test-neutral-run-v1",
            "baseline_indicator": self.baseline,
            "contexts": [self.context],
            "indicators": self.indicators,
            "indicator_markers": {
                "GDP-Growth": ["GDP Growth"],
                "Treasury-Yields": ["Treasury Yields"],
            },
        }
        _write_json(self.roster_path, self.roster)

        self.blocks = {
            "GDP-Growth": self._block(
                "GDP-Growth",
                "GDP Growth",
                "source economic evidence here",
            ),
            "Treasury-Yields": self._block(
                "Treasury-Yields",
                "Treasury Yields",
                "source financial evidence here",
            ),
        }
        self.neutral_blocks = {
            "GDP-Growth": self._block(
                "GDP-Growth",
                "GDP Growth",
                "neutral withheld content here",
            ),
            "Treasury-Yields": self._block(
                "Treasury-Yields",
                "Treasury Yields",
                "neutral withheld content here",
            ),
        }
        parts = ["Prompt prefix\n"]
        self.spans = {}
        for index, indicator in enumerate(self.indicators):
            parts.append(f"--- LOO-BLOCK-SLOT {index:03d} ---\n")
            start = sum(len(part) for part in parts)
            parts.append(self.blocks[indicator])
            end = sum(len(part) for part in parts)
            self.spans[indicator] = (start, end)
        parts.append("--- END-LOO-BLOCK-SLOTS ---\nPrompt suffix\n")
        self.full_prompt = "".join(parts)

        self.neutral_prompts = {}
        for indicator in self.indicators:
            start, end = self.spans[indicator]
            self.neutral_prompts[indicator] = (
                self.full_prompt[:start]
                + self.neutral_blocks[indicator]
                + self.full_prompt[end:]
            )

        baseline_path = (
            self.prompt_dir
            / f"{self.baseline}_masked_{self.context}.jsonl"
        )
        _write_jsonl(
            baseline_path,
            [self._row(self.baseline, "full", self.full_prompt)],
        )
        for indicator in self.indicators:
            _write_jsonl(
                self.prompt_dir / f"{indicator}_masked_{self.context}.jsonl",
                [
                    self._row(
                        indicator,
                        "neutral",
                        self.neutral_prompts[indicator],
                    )
                ],
            )

        self.manifest = {
            "schema_version": "loo-neutral-intervention-v1",
            "population_id": "formal-test-population",
            "population_context": self.context,
            "roster_id": self.roster["roster_id"],
            "roster_file_sha256": sha256_file(self.roster_path),
            "baseline_indicator": self.baseline,
            "indicators": self.indicators,
            "contexts": [self.context],
            "tokenizer_artifact": {"sha256": "b" * 64},
            "replacements": [
                self._proof(indicator)
                for indicator in self.indicators
            ],
        }

    @staticmethod
    def _block(indicator: str, label: str, body: str) -> str:
        return (
            f"<<<LOO-INDICATOR-BLOCK:{indicator} | {label}>>>\n"
            f"Indicator: {label}\n"
            "Analysis:\n"
            f"{body}\n"
            f"<<<END-LOO-INDICATOR-BLOCK:{indicator}>>>\n"
        )

    def _row(self, indicator: str, arm: str, prompt: str) -> dict:
        return {
            "sample_id": self.sample_id,
            "meeting_date": self.meeting_date,
            "section_name": self.section,
            "section_family": self.section,
            "indicator": indicator,
            "evaluation_context": self.context,
            "arm": arm,
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
        }

    def _proof(self, indicator: str) -> dict:
        start, source_end = self.spans[indicator]
        neutral_block = self.neutral_blocks[indicator]
        neutral_end = start + len(neutral_block)
        neutral_prompt = self.neutral_prompts[indicator]
        source_prefix = self.full_prompt[:start]
        source_suffix = self.full_prompt[source_end:]
        tokenizer = WhitespaceTokenizer()
        return {
            "sample_id": self.sample_id,
            "meeting_date": self.meeting_date,
            "section_family": self.section,
            "indicator": indicator,
            "full_prompt_sha256": sha256_text(self.full_prompt),
            "neutral_prompt_sha256": sha256_text(neutral_prompt),
            "source_block_sha256": sha256_text(self.blocks[indicator]),
            "neutral_block_sha256": sha256_text(neutral_block),
            "replacement_start": start,
            "source_replacement_end": source_end,
            "neutral_replacement_end": neutral_end,
            "prefix_sha256": sha256_text(source_prefix),
            "suffix_sha256": sha256_text(source_suffix),
            "source_block_token_count_no_special_tokens": len(
                tokenizer.encode(self.blocks[indicator])
            ),
            "neutral_block_token_count_no_special_tokens": len(
                tokenizer.encode(neutral_block)
            ),
            "source_full_prompt_token_count_no_special_tokens": len(
                tokenizer.encode(self.full_prompt)
            ),
            "neutral_full_prompt_token_count_no_special_tokens": len(
                tokenizer.encode(neutral_prompt)
            ),
            "label_and_delimiters_preserved": True,
            "prefix_suffix_unchanged": True,
            "token_length_matched": True,
        }

    def validate(self, manifest=None):
        return validate_neutral_intervention_manifest(
            manifest=self.manifest if manifest is None else manifest,
            input_dir=self.prompt_dir,
            roster_file=self.roster_path,
            tokenizer=WhitespaceTokenizer(),
            tokenizer_artifact_sha256="b" * 64,
        )


class TestNeutralInterventionValidator(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = NeutralFixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_validates_and_normalizes_generation_ready_manifest(self):
        normalized = self.fixture.validate()

        self.assertEqual(normalized["audit"]["status"], "validated")
        self.assertEqual(normalized["audit"]["input_artifact_count"], 3)
        self.assertEqual(normalized["audit"]["replacement_count"], 2)
        self.assertEqual(len(normalized["input_artifacts"]), 3)
        self.assertEqual(
            {
                (entry["indicator"], entry["arm"])
                for entry in normalized["input_artifacts"]
            },
            {
                ("None", "full"),
                ("GDP-Growth", "neutral"),
                ("Treasury-Yields", "neutral"),
            },
        )
        self.assertTrue(
            all(proof["validated"] for proof in normalized["replacements"])
        )

        # A frozen normalized manifest can be validated again.
        revalidated = self.fixture.validate(normalized)
        self.assertEqual(revalidated, normalized)

    def test_rejects_incomplete_artifact_or_row_coverage(self):
        missing_path = (
            self.fixture.prompt_dir
            / f"Treasury-Yields_masked_{self.fixture.context}.jsonl"
        )
        missing_path.unlink()
        with self.assertRaisesRegex(ValueError, "exact roster universe"):
            self.fixture.validate()

    def test_rejects_row_and_proof_hash_tampering(self):
        with self.subTest("row hash"):
            path = (
                self.fixture.prompt_dir
                / f"GDP-Growth_masked_{self.fixture.context}.jsonl"
            )
            row = json.loads(path.read_text(encoding="utf-8"))
            row["prompt_sha256"] = "0" * 64
            _write_jsonl(path, [row])
            with self.assertRaisesRegex(ValueError, "row hash mismatch"):
                self.fixture.validate()

        # Recreate the fixture in a clean directory for proof tampering.
        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = NeutralFixture(Path(self.temporary.name))
        manifest = json.loads(json.dumps(self.fixture.manifest))
        manifest["replacements"][0]["source_block_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "block/prefix/suffix hash mismatch"):
            self.fixture.validate(manifest)

    def test_rejects_changed_prefix_or_indicator_frame(self):
        indicator = "GDP-Growth"
        path = (
            self.fixture.prompt_dir
            / f"{indicator}_masked_{self.fixture.context}.jsonl"
        )
        row = json.loads(path.read_text(encoding="utf-8"))
        row["prompt"] = "Changed " + row["prompt"][8:]
        row["prompt_sha256"] = sha256_text(row["prompt"])
        _write_jsonl(path, [row])
        manifest = json.loads(json.dumps(self.fixture.manifest))
        manifest["replacements"][0]["neutral_prompt_sha256"] = row["prompt_sha256"]
        with self.assertRaisesRegex(ValueError, "prefix/suffix changed"):
            self.fixture.validate(manifest)

    def test_rejects_block_or_full_prompt_token_count_mismatch(self):
        indicator = "GDP-Growth"
        path = (
            self.fixture.prompt_dir
            / f"{indicator}_masked_{self.fixture.context}.jsonl"
        )
        row = json.loads(path.read_text(encoding="utf-8"))
        original_neutral = self.fixture.neutral_blocks[indicator]
        longer_neutral = original_neutral.replace(
            "neutral withheld content here",
            "neutral withheld additional content here",
        )
        start, source_end = self.fixture.spans[indicator]
        row["prompt"] = (
            self.fixture.full_prompt[:start]
            + longer_neutral
            + self.fixture.full_prompt[source_end:]
        )
        row["prompt_sha256"] = sha256_text(row["prompt"])
        _write_jsonl(path, [row])

        manifest = json.loads(json.dumps(self.fixture.manifest))
        proof = manifest["replacements"][0]
        proof["neutral_prompt_sha256"] = row["prompt_sha256"]
        proof["neutral_block_sha256"] = sha256_text(longer_neutral)
        proof["neutral_replacement_end"] = start + len(longer_neutral)
        proof["neutral_block_token_count_no_special_tokens"] += 1
        proof["neutral_full_prompt_token_count_no_special_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "token-count mismatch"):
            self.fixture.validate(manifest)

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("generate_new_response.generate_responses")
    def test_mask_generation_accepts_only_validated_neutral_prompts(
        self,
        mock_generate_responses,
        mock_tokenizer_loader,
    ):
        mock_tokenizer_loader.return_value = WhitespaceTokenizer()
        mock_generate_responses.side_effect = (
            lambda prompts, model_path, **kwargs: [
                f"generated {index}" for index, _ in enumerate(prompts)
            ]
        )
        root = self.fixture.root
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        tokenizer_hash = fingerprint_artifact_path(model)["sha256"]
        self.fixture.manifest["tokenizer_artifact"]["sha256"] = tokenizer_hash

        neutral_manifest_path = root / "neutral_intervention_manifest.json"
        _write_json(neutral_manifest_path, self.fixture.manifest)
        ledger_path = root / "prompt_ledger.jsonl"
        ledger_path.write_text('{"status":"fixture"}\n', encoding="utf-8")
        artifacts = []
        for path in sorted(self.fixture.prompt_dir.glob("*.jsonl")):
            indicator = path.stem.split("_masked_", 1)[0]
            artifacts.append(
                {
                    "relative_path": f"neutral/{path.name}",
                    "arm": (
                        "full_export"
                        if indicator == self.fixture.baseline
                        else "neutral"
                    ),
                    "indicator": indicator,
                    "context": self.fixture.context,
                    "row_count": 1,
                    "sha256": sha256_file(path),
                }
            )
        prompt_manifest_path = root / "prompt_manifest.json"
        _write_json(
            prompt_manifest_path,
            {
                "schema_version": "loo-prompt-manifest-v1",
                "population_id": "formal-test-population",
                "population_context": self.fixture.context,
                "artifacts": artifacts,
                "source_artifacts": [],
                "run_intervention_roster": {
                    "relative_path": self.fixture.roster_path.name,
                    "sha256": sha256_file(self.fixture.roster_path),
                },
                "prompt_ledger": {
                    "relative_path": ledger_path.name,
                    "sha256": sha256_file(ledger_path),
                },
                "neutral_intervention_manifest": {
                    "relative_path": neutral_manifest_path.name,
                    "sha256": sha256_file(neutral_manifest_path),
                },
            },
        )
        output = root / "generated"

        run_mask_generation(
            input_folder=str(self.fixture.prompt_dir),
            model_path=str(model),
            tokenizer_path=str(model),
            simulation_step=1,
            output_dir=str(output),
            roster_file=str(self.fixture.roster_path),
            batch_size=1,
            seed=20260728,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=128,
            masking_strategy="indicator_block_neutral_replacement",
            prompt_manifest_file=str(prompt_manifest_path),
            intervention_manifest_file=str(neutral_manifest_path),
        )

        generation_manifest = json.loads(
            (output / "generation_manifest.json").read_text(encoding="utf-8")
        )
        normalized_intervention = json.loads(
            (output / "intervention_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            generation_manifest["masking_strategy"],
            "indicator_block_neutral_replacement",
        )
        self.assertEqual(
            generation_manifest["intervention_manifest"]["schema_version"],
            "loo-neutral-intervention-v1",
        )
        self.assertEqual(
            normalized_intervention["audit"]["status"],
            "validated",
        )

    def test_rejects_non_distinct_neutral_prompts(self):
        first = "GDP-Growth"
        second = "Treasury-Yields"
        first_path = (
            self.fixture.prompt_dir
            / f"{first}_masked_{self.fixture.context}.jsonl"
        )
        second_path = (
            self.fixture.prompt_dir
            / f"{second}_masked_{self.fixture.context}.jsonl"
        )
        first_row = json.loads(first_path.read_text(encoding="utf-8"))
        second_row = dict(
            first_row,
            indicator=second,
        )
        _write_jsonl(second_path, [second_row])
        with self.assertRaisesRegex(ValueError, "not distinct"):
            self.fixture.validate()

    def test_validates_declared_input_artifact_hashes(self):
        normalized = self.fixture.validate()
        normalized["input_artifacts"][0]["source_file_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "input_artifact mismatch"):
            self.fixture.validate(normalized)


if __name__ == "__main__":
    unittest.main()
