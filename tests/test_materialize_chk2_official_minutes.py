from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import jobs.generation.materialize_chk2_official_minutes as materializer
from jobs.generation.materialize_chk2_official_minutes import (
    BuildPolicy,
    OfficialMinutesDataError,
    materialize_release,
)


class _WhitespaceTokenizer:
    eos_token = "<eos>"

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))

    @staticmethod
    def apply_chat_template(
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(message["content"] for message in messages) + "\n<think>\n"

    @staticmethod
    def __call__(*, text: str) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_row(
    sample_id: str,
    *,
    split: str = "train",
    meeting_date: str = "2020-01-29",
    official: str | None = None,
) -> dict:
    official = official or (
        "Economic activity expanded at a moderate pace, while labor market "
        "conditions remained firm and inflation pressures were described as "
        "subdued over the period under review."
    )
    return {
        "sample_id": sample_id,
        "split": split,
        "meeting_date": meeting_date,
        "section_name": "Staff Review of the Economic Situation",
        "topic": "GDP Growth",
        "source_row_index": 7,
        "quality_flags": [],
        "raw_analysis": (
            "Economic activity expanded at a moderate pace. Labor market "
            "conditions remained firm, while inflation pressures were subdued."
        ),
        "reference_excerpt": official,
        "teacher_rewrite_reasoning": (
            "Retain the moderate expansion in activity, the firm labor market, "
            "and the subdued inflation assessment. Use neutral institutional "
            "wording and connect the observations in one concise paragraph."
        ),
        "teacher_rewrite_response": "This synthetic answer must not be supervision.",
        "teacher_model": "teacher-test",
        "teacher_status": "success",
    }


def _policy() -> BuildPolicy:
    return BuildPolicy(
        min_reasoning_tokens=1,
        max_reasoning_tokens=500,
        min_target_words=5,
        max_target_words=100,
        min_lexical_jaccard=0.0,
        prompt_token_limit=1000,
        total_token_limit=2000,
        require_official_source_match=False,
    )


def _prepare_source(root: Path, rows: dict[str, list[dict]]) -> None:
    for split in ("train", "eval", "test"):
        _write_jsonl(root / f"{split}_manifest.jsonl", rows.get(split, []))


def test_release_uses_official_minutes_not_teacher_synthetic_answer(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "release"
    row = _source_row("row-1")
    _prepare_source(source, {"train": [row]})

    handoff = materialize_release(
        source_root=source,
        output_root=output,
        tokenizer=_WhitespaceTokenizer(),
        policy=_policy(),
    )

    training = json.loads(
        (output / "minutes_alignment/train.jsonl").read_text(encoding="utf-8")
    )
    assert set(training) == {"prompt", "response"}
    reasoning, final_answer = training["response"].split("\n</think>\n")
    assert reasoning
    assert final_answer == row["reference_excerpt"]
    assert row["teacher_rewrite_response"] not in training["response"]
    assert row["reference_excerpt"] not in training["prompt"]

    manifest = json.loads(
        (output / "minutes_alignment/manifests/train.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["target_source_field"] == "reference_excerpt"
    assert manifest["target_is_teacher_synthetic_rewrite"] is False
    assert handoff["split_counts"] == {"test": 0, "train": 1, "validation": 0}
    assert handoff["training_ready"] is False
    assert json.loads((output / "release_manifest.json").read_text())["training_ready"] is False
    assert json.loads((output / "audits/data_quality.json").read_text())["training_ready"] is False


def test_unsupported_official_number_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "release"
    accepted = _source_row("accepted")
    rejected = _source_row(
        "rejected",
        official=(
            "Economic activity expanded at a moderate pace and inflation was "
            "reported at 5 percent, while labor market conditions remained firm."
        ),
    )
    rejected["source_row_index"] = 8
    _prepare_source(source, {"train": [accepted, rejected]})

    materialize_release(
        source_root=source,
        output_root=output,
        tokenizer=_WhitespaceTokenizer(),
        policy=_policy(),
    )

    training_rows = (output / "minutes_alignment/train.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(training_rows) == 1
    rejection = json.loads(
        (output / "rejections/train.jsonl").read_text(encoding="utf-8")
    )
    assert "unsupported_target_numbers" in rejection["rejection_reasons"]


def test_legacy_transport_and_draft_tail_are_removed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "release"
    row = _source_row("row-with-meta")
    row["teacher_rewrite_reasoning"] = (
        "The user wants me to transform the Raw Analysis into one formal paragraph. "
        "Do not use lists or headings. No new facts. "
        "Retain the moderate expansion, firm labor market, and subdued inflation; "
        "connect them with neutral institutional wording. Let me draft: This is a "
        "synthetic paragraph that must not survive in the reasoning supervision."
    )
    _prepare_source(source, {"train": [row]})

    materialize_release(
        source_root=source,
        output_root=output,
        tokenizer=_WhitespaceTokenizer(),
        policy=_policy(),
    )

    training = json.loads(
        (output / "minutes_alignment/train.jsonl").read_text(encoding="utf-8")
    )
    reasoning = training["response"].split("\n</think>\n", 1)[0]
    assert "The user" not in reasoning
    assert "Raw Analysis" not in reasoning
    assert "Do not use lists" not in reasoning
    assert "No new facts" not in reasoning
    assert "synthetic paragraph" not in reasoning
    assert reasoning.startswith("Retain the moderate expansion")
    manifest = json.loads(
        (output / "minutes_alignment/manifests/train.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["diagnostics"]["reasoning_draft_tail_removed"] is True
    assert manifest["diagnostics"]["reasoning_meta_categories"] == []


def test_official_target_repairs_enumerated_mojibake(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "release"
    row = _source_row(
        "row-with-mojibake",
        official=(
            "Economic activity expanded at a moderate paceâwhile labor market "
            "conditions remained firm and inflation pressures were described as "
            "subdued over the period under review."
        ),
    )
    _prepare_source(source, {"train": [row]})

    materialize_release(
        source_root=source,
        output_root=output,
        tokenizer=_WhitespaceTokenizer(),
        policy=_policy(),
    )

    training = json.loads(
        (output / "minutes_alignment/train.jsonl").read_text(encoding="utf-8")
    )
    assert "â" not in training["response"]
    assert "pace—while" in training["response"]
    manifest = json.loads(
        (output / "minutes_alignment/manifests/train.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["official_minutes_normalization_repairs"]


@pytest.mark.parametrize(
    "official",
    [
        (
            "Committee Policy ActionMembers agreed that it was appropriate to "
            "maintain the target range for the federal funds rate over the coming "
            "period in light of the outlook."
        ),
        (
            "Participants agreed to leave the target range for the federal funds "
            "rate unchanged while continuing to assess incoming information."
        ),
    ],
)
def test_policy_and_glued_heading_targets_are_rejected(
    tmp_path: Path, official: str
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "release"
    accepted = _source_row("accepted")
    rejected = _source_row("policy", official=official)
    rejected["source_row_index"] = 9
    _prepare_source(source, {"train": [accepted, rejected]})

    materialize_release(
        source_root=source,
        output_root=output,
        tokenizer=_WhitespaceTokenizer(),
        policy=_policy(),
    )

    rejection = json.loads(
        (output / "rejections/train.jsonl").read_text(encoding="utf-8")
    )
    assert "administrative_or_policy_target" in rejection["rejection_reasons"]


def test_official_target_must_match_exactly_one_labeled_csv_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "release"
    accepted = _source_row("accepted")
    fabricated = _source_row(
        "fabricated",
        official=(
            "Economic activity expanded at a moderate pace, while labor market "
            "conditions remained firm and inflation pressures were notably "
            "subdued across the period."
        ),
    )
    fabricated["source_row_index"] = 10
    accepted_target = accepted["reference_excerpt"]

    def fake_source_matches(meeting_date: str, official: str) -> list[dict]:
        del meeting_date
        if official != accepted_target:
            return []
        return [
            {
                "path": "official.csv",
                "line_id": "7",
                "section_name": "Staff Review",
                "raw_text_sha256": "a" * 64,
                "projected_text_sha256": "b" * 64,
                "normalization_repairs": [],
            }
        ]

    monkeypatch.setattr(
        materializer,
        "_official_source_matches",
        fake_source_matches,
    )
    _prepare_source(source, {"train": [accepted, fabricated]})

    materialize_release(
        source_root=source,
        output_root=output,
        tokenizer=_WhitespaceTokenizer(),
        policy=replace(_policy(), require_official_source_match=True),
    )

    rejection = json.loads(
        (output / "rejections/train.jsonl").read_text(encoding="utf-8")
    )
    assert "official_source_exact_match_count_not_one" in rejection[
        "rejection_reasons"
    ]


def test_meeting_overlap_across_splits_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "release"
    train = _source_row("train-row", split="train")
    evaluation = _source_row("eval-row", split="eval")
    _prepare_source(source, {"train": [train], "eval": [evaluation]})

    with pytest.raises(OfficialMinutesDataError, match="meeting split overlap"):
        materialize_release(
            source_root=source,
            output_root=output,
            tokenizer=_WhitespaceTokenizer(),
            policy=_policy(),
        )
