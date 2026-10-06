from __future__ import annotations

import copy
from pathlib import Path

import pytest

from jobs.eval import (
    eval_chk3_beta_core8_sentiment_stochastic_schedule_v1 as sentiment,
)
from open_r1.validator.loo_generation_spec import seal_manifest


class _FakeTokenizer:
    model_max_length = 512

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(int(text))) if text else []

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        assert pair is False
        return 2

    def build_inputs_with_special_tokens(self, values: list[int]) -> list[int]:
        return [101, *values, 102]


def test_frozen_model_contracts_and_label_orders() -> None:
    distil = sentiment.MODEL_CONTRACTS[sentiment.DISTIL_BACKEND]
    finbert = sentiment.MODEL_CONTRACTS[sentiment.FINBERT_BACKEND]
    assert distil.revision == "9c061b4ce901418ff7839e9d89ca07156125d201"
    assert distil.labels == ("dovish", "hawkish", "neutral")
    assert (distil.positive_index, distil.negative_index) == (1, 0)
    assert finbert.revision == "4556d13015211d73dccd3fdd39d39232506f3e43"
    assert finbert.labels == ("positive", "negative", "neutral")
    assert (finbert.positive_index, finbert.negative_index) == (0, 1)
    assert distil.construct != finbert.construct


def test_510_128_windows_cover_tokens_once_after_correction() -> None:
    specs = sentiment._window_specs(_FakeTokenizer(), "523")
    assert [(spec.token_start, spec.token_end) for spec in specs] == [
        (0, 510),
        (382, 523),
    ]
    assert all(len(spec.input_ids) <= 512 for spec in specs)
    assert sum(spec.aggregation_weight for spec in specs) == pytest.approx(523.0)
    # The 128-token overlap contributes a total mass of 64 to each window.
    assert specs[0].aggregation_weight == pytest.approx(446.0)
    assert specs[1].aggregation_weight == pytest.approx(77.0)
    assert sentiment._window_specs(_FakeTokenizer(), "") == []


def test_lexicon_is_phrase_level_and_normalized_by_words() -> None:
    text = "A hawkish Committee could raise rates; policy accommodation may ease."
    result = sentiment._lexicon_score(text)
    assert result["hawkish_count"] == 2
    assert result["dovish_count"] == 2
    assert result["score"] == 0.0
    assert result["token_count"] == 10
    assert sentiment._lexicon_score("")["score"] is None


def _patch_small_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sentiment.data_contract, "CORE_TOPICS", ("topic-a", "topic-b"))
    monkeypatch.setattr(sentiment.data_contract, "EXPECTED_MEETINGS", 1)
    monkeypatch.setattr(sentiment, "TOPIC_COUNT", 2)
    monkeypatch.setattr(sentiment, "REPLICATE_COUNT", 1)
    monkeypatch.setattr(sentiment, "EXPECTED_REFERENCE_TOPICS", 2)
    monkeypatch.setattr(sentiment, "EXPECTED_GENERATED_TOPICS_PER_ARM", 2)
    monkeypatch.setattr(sentiment, "EXPECTED_TEXTS", 8)
    monkeypatch.setattr(sentiment, "EXPECTED_MEETING_ROWS", 4)


def _inventory_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in sentiment.ARM_ORDER:
        replicate_id = None if arm == "reference" else 0
        for topic_order, topic in enumerate(("topic-a", "topic-b")):
            text = "hawkish tighten" if topic_order == 0 else "dovish ease"
            if arm == "chk3" and topic_order == 1:
                text = ""
            text_id = (
                f"reference::sample-{topic_order}"
                if arm == "reference"
                else f"{arm}::sample-{topic_order}::replicate-00"
            )
            rows.append(
                {
                    "schema_version": sentiment.INVENTORY_SCHEMA,
                    "text_id": text_id,
                    "arm": arm,
                    "sample_id": f"sample-{topic_order}",
                    "meeting_id": "2000-01-01",
                    "meeting_date": "2000-01-01",
                    "meeting_start_date": "1999-12-31",
                    "era": "test-era",
                    "role": "external_holdout",
                    "topic": topic,
                    "topic_order": topic_order,
                    "replicate_id": replicate_id,
                    "replicate_seed": (
                        None
                        if replicate_id is None
                        else sentiment.preparation.REPLICATE_SEEDS[0]
                    ),
                    "cp318_selection_exposed": False,
                    "text": text,
                    "text_sha256": sentiment.sha256_text(text),
                    "empty": not bool(text.strip()),
                    "source_generation_line_number": topic_order + 1,
                    "source_generation_model_id": "chk0" if arm == "reference" else arm,
                    "source_generation_completion_sha256": "a" * 64,
                    "source_reference_minutes_sha256": "b" * 64,
                    "source_answer_sha256": "c" * 64,
                }
            )
    return rows


def _small_inventory_manifest(tmp_path: Path) -> Path:
    directory = tmp_path / "inventory"
    directory.mkdir()
    rows_path = directory / "text_inventory.v1.jsonl"
    with sentiment._JsonlWriter(rows_path) as writer:
        for row in _inventory_rows():
            writer.write(row)
    manifest = seal_manifest(
        {
            "schema_version": sentiment.INVENTORY_MANIFEST_SCHEMA,
            "status": "complete",
            "immutable": True,
            "evaluation_id": sentiment.EVALUATION_ID,
            "text_inventory": sentiment._binding(rows_path, rows=8),
            "implementation": sentiment._implementation_binding(),
        }
    )
    path = directory / "manifest.json"
    sentiment._write_readonly_json(path, manifest)
    return path


def test_lexicon_meeting_loader_rebuilds_and_preserves_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_small_contract(monkeypatch)
    inventory = _small_inventory_manifest(tmp_path)
    output = tmp_path / "lexicon"
    sentiment.score_lexicon(inventory_manifest=inventory, output_dir=output)
    loaded = sentiment.load_and_validate_meeting_scores(output / "manifest.json")
    rows = loaded["meeting_scores"]
    assert len(rows) == 4
    chk3 = next(row for row in rows if row["arm"] == "chk3")
    assert chk3["score"] is None
    assert chk3["complete_core8"] is False
    assert chk3["missing_topics"] == ["topic-b"]
    assert chk3["neutral_imputed_score"] != 0.0
    assert chk3["cp318_selection_exposed"] is False
    assert set(chk3["source_hashes"]) == {
        "topic_text_sha256s",
        "topic_score_row_sha256s",
    }


def test_meeting_loader_rejects_topic_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_small_contract(monkeypatch)
    inventory = _small_inventory_manifest(tmp_path)
    output = tmp_path / "lexicon"
    sentiment.score_lexicon(inventory_manifest=inventory, output_dir=output)
    path = output / "topic_scores.jsonl"
    path.chmod(0o644)
    rows = sentiment._read_jsonl(path)
    altered = copy.deepcopy(rows)
    altered[0]["score"] = -0.75
    path.write_text(
        "".join(sentiment._canonical(row) + "\n" for row in altered),
        encoding="utf-8",
    )
    with pytest.raises(sentiment.SentimentScoringError, match="binding drift"):
        sentiment.load_and_validate_meeting_scores(output / "manifest.json")


def test_meeting_aggregation_rejects_mixed_core8_cp318_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_small_contract(monkeypatch)
    inventory = _small_inventory_manifest(tmp_path)
    loaded = sentiment.load_and_validate_inventory(inventory)
    rows = loaded["rows"]
    topic_rows = []
    for source in rows:
        topic_rows.append(
            {
                **sentiment._topic_base(source, backend_id=sentiment.LEXICON_BACKEND),
                "score": 0.0,
            }
        )
    topic_rows[1]["cp318_selection_exposed"] = True
    with pytest.raises(
        sentiment.SentimentScoringError, match="cp318_selection_exposed"
    ):
        sentiment._aggregate_meeting_rows(
            topic_rows, backend_id=sentiment.LEXICON_BACKEND
        )
