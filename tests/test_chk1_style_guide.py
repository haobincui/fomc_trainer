from __future__ import annotations

import copy
import hashlib
import json
import re

import pytest

from jobs.retrain_v2.chk1.style_guide import (
    StyleGuideError,
    abstract_minutes_style,
    abstract_minutes_text,
    build_style_guide,
    build_style_guide_from_jsonl,
    stable_section_style_id,
    verify_style_guide_artifact,
)


def _row(
    sample_id: str,
    *,
    split: str = "train",
    meeting_date: str = "2020-01-29",
    section_name: str = "Staff Review of the Economic Situation",
    source_row_index: int = 1,
    excerpt: str = (
        "Incoming information appeared mixed. However, several indicators "
        "suggested that uncertainty remained elevated."
    ),
) -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "meeting_date": meeting_date,
        "section_name": section_name,
        "source_row_index": source_row_index,
        "reference_excerpt": excerpt,
    }


def test_abstraction_uses_only_controlled_atoms_and_excludes_factual_sentences() -> None:
    source = (
        "On January 31, 2024, Chair Powell stated that inflation was 3.2 percent. "
        "The Committee voted unanimously to maintain the target range. "
        "Participants generally noted uncertainty; however, conditions appeared "
        "mixed and might remain so."
    )

    signature = abstract_minutes_style(source)
    abstracted = abstract_minutes_text(source)

    assert abstracted == signature["abstracted_text"]
    assert signature["source_sentence_count"] == 3
    assert signature["usable_sentence_count"] == 1
    assert signature["excluded_sentence_counts"] == {
        "person_reference": 1,
        "policy_outcome": 1,
    }
    assert {"contrast", "uncertainty", "source_attribution"}.issubset(
        signature["moves"]
    )
    assert {"however", "appeared", "might"}.issubset(
        signature["lexical_markers"]
    )
    assert re.fullmatch(r"[a-z_; ]+", abstracted)
    for prohibited in ("powell", "january", "2024", "3.2", "maintain", "target"):
        assert prohibited not in abstracted.casefold()


def test_section_style_id_is_opaque_stable_and_alias_aware() -> None:
    canonical = "Participants' Views on Current Conditions and the Economic Outlook"
    variant = "  participants’ VIEW on current conditions and the economic outlook  "
    canonical_id = stable_section_style_id(canonical)

    assert stable_section_style_id(variant) == canonical_id
    assert re.fullmatch(r"section-style-v1-[0-9a-f]{20}", canonical_id)
    assert "participants" not in canonical_id
    assert stable_section_style_id("Staff Review of the Financial Situation") != canonical_id

    # Date-specific administrative headings must not become meeting proxies.
    assert stable_section_style_id(
        "Videoconference meeting of October 4, 2019"
    ) == stable_section_style_id("September 21–22, 2021")
    assert stable_section_style_id(",") == stable_section_style_id("-")


@pytest.mark.parametrize("held_out_split", ["eval", "test", "validation", "TRAIN"])
def test_builder_rejects_every_non_train_split(held_out_split: str) -> None:
    with pytest.raises(StyleGuideError, match="only accepts split='train'"):
        build_style_guide([_row("held-out", split=held_out_split)])


def test_builder_rejects_mixed_splits_and_duplicate_sample_ids() -> None:
    with pytest.raises(StyleGuideError, match="only accepts split='train'"):
        build_style_guide(
            [
                _row("train-row"),
                _row("test-row", split="test", source_row_index=2),
            ]
        )

    with pytest.raises(StyleGuideError, match="duplicate sample_id"):
        build_style_guide(
            [
                _row("duplicate", source_row_index=1),
                _row("duplicate", source_row_index=2),
            ]
        )


def test_builder_requires_existing_minutes_target_schema() -> None:
    missing_reference = _row("missing-reference")
    missing_reference["response"] = missing_reference.pop("reference_excerpt")
    with pytest.raises(StyleGuideError, match="reference_excerpt"):
        build_style_guide([missing_reference])

    missing_split = _row("missing-split")
    del missing_split["split"]
    with pytest.raises(StyleGuideError, match="split"):
        build_style_guide([missing_split])


def test_guides_are_fact_free_and_hashes_are_order_independent() -> None:
    rows = [
        _row(
            "economic-a",
            meeting_date="2020-01-29",
            source_row_index=11,
            excerpt=(
                "Payroll employment declined by 225,000 in January. Although the "
                "reading was weak, staff judged that uncertainty remained elevated."
            ),
        ),
        _row(
            "economic-b",
            meeting_date="2020-03-15",
            source_row_index=12,
            excerpt=(
                "On March 15, activity appeared softer. However, several indicators "
                "suggested that conditions could stabilize."
            ),
        ),
        _row(
            "financial-a",
            meeting_date="2020-04-29",
            section_name="Staff Review of the Financial Situation",
            source_row_index=13,
            excerpt=(
                "Market participants reportedly observed tighter conditions, while "
                "some measures remained volatile."
            ),
        ),
    ]

    artifact = build_style_guide(rows, source_id="unit-test-train-minutes")
    reordered = build_style_guide(
        list(reversed(rows)), source_id="unit-test-train-minutes"
    )

    assert artifact == reordered
    assert verify_style_guide_artifact(artifact) is True
    assert artifact["provenance"]["corpus_row_count"] == 3
    assert artifact["provenance"]["section_style_count"] == 2
    assert artifact["provenance"]["corpus_sha256"] == reordered["provenance"][
        "corpus_sha256"
    ]
    assert artifact["style_guide_sha256"] == reordered["style_guide_sha256"]

    serialized_styles = json.dumps(artifact["styles"], sort_keys=True).casefold()
    for fact in ("225,000", "january", "march 15", "2020-03-15", "powell"):
        assert fact not in serialized_styles
    for style in artifact["styles"]:
        assert not re.search(r"\d", style["guide_text"])
        assert style["guide_text_sha256"] == hashlib.sha256(
            style["guide_text"].encode("utf-8")
        ).hexdigest()
        assert "section_name" not in style

    changed_rows = copy.deepcopy(rows)
    changed_rows[0]["reference_excerpt"] += " Incoming evidence seemed uneven."
    changed = build_style_guide(
        changed_rows, source_id="unit-test-train-minutes"
    )
    assert changed["provenance"]["corpus_sha256"] != artifact["provenance"][
        "corpus_sha256"
    ]
    assert changed["style_guide_sha256"] != artifact["style_guide_sha256"]


def test_policy_only_excerpt_gets_safe_fallback_not_a_sanitized_example() -> None:
    artifact = build_style_guide(
        [
            _row(
                "policy-only",
                section_name="Monetary Policy Discussion",
                excerpt=(
                    "By unanimous vote, the Committee decided to lower the target "
                    "range by 50 basis points."
                ),
            )
        ]
    )
    style = artifact["styles"][0]
    assert style["usable_row_count"] == 0
    assert style["excluded_sentence_counts"] == {"policy_outcome": 1}
    assert style["style_signature"]["moves"] == []
    assert "lower" not in style["guide_text"].casefold()
    assert "basis" not in style["guide_text"].casefold()
    assert not re.search(r"\d", style["guide_text"])


def test_guide_allows_only_fact_card_values_and_gates_causal_language() -> None:
    artifact = build_style_guide(
        [
            _row(
                "causal-style",
                excerpt=(
                    "Conditions softened, reflecting weaker demand. However, incoming "
                    "information appeared mixed."
                ),
            )
        ]
    )
    guide = artifact["styles"][0]["guide_text"]
    lowered = guide.casefold()

    assert "this style guidance supplies no facts" in lowered
    assert (
        "use numerical values and dates only when they appear in fact-card evidence"
        in lowered
    )
    assert "only when the fact-card evidence explicitly states the relationship" in lowered
    assert "otherwise describe association or sequence without causation" in lowered
    assert "do not introduce named individuals" in lowered
    assert not re.search(r"\d", guide)


def test_jsonl_reader_pins_raw_source_and_rejects_held_out_paths(tmp_path) -> None:
    rows = [
        _row("row-a", source_row_index=21),
        _row(
            "row-b",
            source_row_index=22,
            meeting_date="2020-03-15",
            section_name="Staff Review of the Financial Situation",
        ),
    ]
    train_path = tmp_path / "train.jsonl"
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    train_path.write_text(payload, encoding="utf-8")
    expected_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    first = build_style_guide_from_jsonl(
        train_path, expected_source_sha256=expected_sha256
    )
    second = build_style_guide_from_jsonl(
        train_path, expected_source_sha256=expected_sha256
    )
    assert first == second
    assert first["provenance"]["source_artifact_sha256"] == expected_sha256
    assert first["provenance"]["source_artifact_bytes"] == len(payload.encode("utf-8"))

    with pytest.raises(StyleGuideError, match="SHA-256 mismatch"):
        build_style_guide_from_jsonl(
            train_path, expected_source_sha256="0" * 64
        )

    held_out_path = tmp_path / "eval.jsonl"
    held_out_path.write_text(payload, encoding="utf-8")
    with pytest.raises(StyleGuideError, match="held-out eval/test source path"):
        build_style_guide_from_jsonl(held_out_path)


def test_verifier_detects_guide_or_provenance_tampering() -> None:
    artifact = build_style_guide([_row("tamper-check")])

    changed_guide = copy.deepcopy(artifact)
    changed_guide["styles"][0]["guide_text"] += " Extra prose."
    with pytest.raises(StyleGuideError, match="guide_text_sha256 mismatch"):
        verify_style_guide_artifact(changed_guide)

    changed_provenance = copy.deepcopy(artifact)
    changed_provenance["provenance"]["corpus_rows"][0]["source_text_sha256"] = (
        "0" * 64
    )
    with pytest.raises(StyleGuideError, match="corpus_sha256 mismatch"):
        verify_style_guide_artifact(changed_provenance)
