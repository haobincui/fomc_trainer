from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from jobs.retrain_v2.chk1.contracts import MANIFEST_SCHEMA_VERSION, sha256_text
from jobs.retrain_v2.chk1.release import (
    QualityThresholds,
    ReleaseValidationError,
    assess_release_quality,
    publish_release,
)


DIGEST = "b" * 64


def _rows(split: str, meeting: str, topic: str) -> tuple[dict, dict]:
    reasoning = f"The {topic} evidence for {split} was reviewed."
    final = f"The available {topic} observations were broadly stable in {split}."
    response = f"{reasoning}\n</think>\n{final}"
    prompt = f"offline fact card for {meeting} {topic} {split}"
    data = f'{{"evidence_id":"{split}-E1","value":"1.0"}}'
    sft = {"prompt": prompt, "response": response, "provided_data": data}
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "sample_id": f"{meeting}::{topic}",
        "meeting_date": meeting,
        "atomic_topic": topic,
        "section_style_id": "participants_views",
        "split": split,
        "cutoff_ts": f"{meeting}T00:00:00Z",
        "evidence_lineage": [{"evidence_id": f"{split}-E1", "source_sha256": DIGEST}],
        "style_guide_sha256": DIGEST,
        "teacher_model_sha256": DIGEST,
        "tokenizer_sha256": DIGEST,
        "generation": {
            "seed": 42,
            "temperature": 0.2,
            "cache_key": DIGEST,
            "generation_provenance_sha256": DIGEST,
            "prompt_template_sha256": DIGEST,
            "generator_tokenizer_sha256": DIGEST,
            "student_tokenizer_sha256": DIGEST,
        },
        "prompt_sha256": sha256_text(prompt),
        "reasoning_sha256": sha256_text(reasoning),
        "final_analysis_sha256": sha256_text(final),
        "response_sha256": sha256_text(response),
        "provided_data_sha256": sha256_text(data),
        "input_truncated": False,
    }
    return sft, manifest


def _complete() -> tuple[dict, dict]:
    inputs = {
        "train": ("2019-01-30", "GDP Growth"),
        "eval": ("2022-01-26", "Unemployment Rate"),
        "test": ("2024-01-31", "Consumer Price Index"),
    }
    sft: dict[str, list[dict]] = {}
    manifests: dict[str, list[dict]] = {}
    for split, (meeting, topic) in inputs.items():
        sft_row, manifest = _rows(split, meeting, topic)
        sft[split] = [sft_row]
        manifests[split] = [manifest]
    return sft, manifests


def _thresholds(
    *,
    exact_meetings: bool = True,
) -> QualityThresholds:
    meetings: dict[str, int | tuple[str, ...]]
    if exact_meetings:
        meetings = {
            "train": ("2019-01-30",),
            "eval": ("2022-01-26",),
            "test": ("2024-01-31",),
        }
    else:
        meetings = {"train": 1, "eval": 1, "test": 1}
    return QualityThresholds(
        expected_meetings=meetings,
        expected_sample_ids={
            "train": ("2019-01-30::GDP Growth",),
            "eval": ("2022-01-26::Unemployment Rate",),
            "test": ("2024-01-31::Consumer Price Index",),
        },
        expected_topics=(
            "GDP Growth",
            "Unemployment Rate",
            "Consumer Price Index",
        ),
    )


def test_quality_uses_automated_gates_and_unique_canonical_keys() -> None:
    sft, manifests = _complete()
    quality, _, _ = assess_release_quality(
        sft_rows=sft,
        manifest_rows=manifests,
        exclusions=[],
        thresholds=_thresholds(),
    )
    assert quality["status"] == "passed"
    assert set(quality["gates"]) == {
        "train_acceptance",
        "per_topic_acceptance",
        "zero_tolerance_exclusions",
    }
    assert "human_audit" not in quality

    manifests["eval"][0]["sample_id"] = manifests["train"][0]["sample_id"]
    with pytest.raises(ReleaseValidationError, match="duplicate sample_id"):
        assess_release_quality(
            sft_rows=sft,
            manifest_rows=manifests,
            exclusions=[],
            thresholds=_thresholds(),
        )


def test_exact_population_meeting_topic_bindings_and_count_compatibility() -> None:
    sft, manifests = _complete()
    exact, _, _ = assess_release_quality(
        sft_rows=sft,
        manifest_rows=manifests,
        exclusions=[],
        thresholds=_thresholds(),
    )
    counted, _, _ = assess_release_quality(
        sft_rows=sft,
        manifest_rows=manifests,
        exclusions=[],
        thresholds=_thresholds(exact_meetings=False),
    )
    assert exact["status"] == "passed"
    assert exact["meeting_binding_modes"] == {
        "train": "exact_membership",
        "eval": "exact_membership",
        "test": "exact_membership",
    }
    assert counted["status"] == "passed"
    assert set(counted["meeting_binding_modes"].values()) == {"count"}

    wrong_meetings = replace(
        _thresholds(),
        expected_meetings={
            "train": ("2019-03-20",),
            "eval": ("2022-01-26",),
            "test": ("2024-01-31",),
        },
    )
    with pytest.raises(ReleaseValidationError, match="meeting membership mismatch"):
        assess_release_quality(
            sft_rows=sft,
            manifest_rows=manifests,
            exclusions=[],
            thresholds=wrong_meetings,
        )

    wrong_population = replace(
        _thresholds(),
        expected_sample_ids={
            "train": ("2019-01-30::GDP Growth", "missing-sample"),
            "eval": ("2022-01-26::Unemployment Rate",),
            "test": ("2024-01-31::Consumer Price Index",),
        },
    )
    with pytest.raises(ReleaseValidationError, match="expected population mismatch"):
        assess_release_quality(
            sft_rows=sft,
            manifest_rows=manifests,
            exclusions=[],
            thresholds=wrong_population,
        )

    wrong_topics = replace(
        _thresholds(),
        expected_topics=("GDP Growth", "Unemployment Rate", "Treasury Yields"),
    )
    with pytest.raises(ReleaseValidationError, match="topic roster mismatch"):
        assess_release_quality(
            sft_rows=sft,
            manifest_rows=manifests,
            exclusions=[],
            thresholds=wrong_topics,
        )


def test_exclusions_cover_population_once_and_zero_acceptance_topic_fails() -> None:
    sft, manifests = _complete()
    excluded_id = "2019-01-30::Treasury Yields"
    thresholds = replace(
        _thresholds(),
        expected_sample_ids={
            "train": ("2019-01-30::GDP Growth", excluded_id),
            "eval": ("2022-01-26::Unemployment Rate",),
            "test": ("2024-01-31::Consumer Price Index",),
        },
        expected_topics=(
            "GDP Growth",
            "Unemployment Rate",
            "Consumer Price Index",
            "Treasury Yields",
        ),
    )
    exclusion = {
        "sample_id": excluded_id,
        "split": "train",
        "meeting_date": "2019-01-30",
        "atomic_topic": "Treasury Yields",
        "reason_code": "critic_rejected",
    }
    quality, _, _ = assess_release_quality(
        sft_rows=sft,
        manifest_rows=manifests,
        exclusions=[exclusion],
        thresholds=thresholds,
    )
    assert quality["status"] == "failed"
    assert quality["gates"]["per_topic_acceptance"] is False
    assert quality["per_topic_acceptance_rates"]["Treasury Yields"] == 0
    assert quality["topics_below_threshold"]["Treasury Yields"] == 0

    with pytest.raises(
        ReleaseValidationError, match="requires non-empty string 'sample_id'"
    ):
        assess_release_quality(
            sft_rows=sft,
            manifest_rows=manifests,
            exclusions=[
                {key: value for key, value in exclusion.items() if key != "sample_id"}
            ],
            thresholds=thresholds,
        )
    with pytest.raises(ReleaseValidationError, match="duplicate excluded sample_id"):
        assess_release_quality(
            sft_rows=sft,
            manifest_rows=manifests,
            exclusions=[exclusion, exclusion],
            thresholds=thresholds,
        )


def test_atomic_release_has_only_minimal_sft_and_handoff_hashes(tmp_path: Path) -> None:
    sft, manifests = _complete()
    handoff_path = publish_release(
        release_root=tmp_path / "releases",
        release_id="release-test",
        sft_rows=sft,
        manifest_rows=manifests,
        exclusions=[],
        thresholds=_thresholds(),
        prompt_template_sha256=DIGEST,
        style_guide_sha256=DIGEST,
        teacher_model_sha256=DIGEST,
        tokenizer_sha256=DIGEST,
        legacy_inventory={"schema_version": "test", "files": []},
        generated_at_utc="2026-08-03T00:00:00Z",
        seal_permissions=False,
    )

    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    quality = json.loads(
        (handoff_path.parent / "audit/quality_report.json").read_text(encoding="utf-8")
    )
    assert handoff["quality_status"] == "passed"
    assert handoff["population_count"] == 3
    assert handoff["split_counts"] == {"train": 1, "eval": 1, "test": 1}
    for field in (
        "population_binding_sha256",
        "split_binding_sha256",
        "topic_binding_sha256",
    ):
        assert len(handoff[field]) == 64
        assert handoff[field] == quality[field]
    row = json.loads(
        (handoff_path.parent / "sft/train.jsonl").read_text(encoding="utf-8")
    )
    assert set(row) == {"prompt", "response", "provided_data"}
    assert "sample_id" not in row
    with pytest.raises(FileExistsError, match="Immutable release"):
        publish_release(
            release_root=tmp_path / "releases",
            release_id="release-test",
            sft_rows=sft,
            manifest_rows=manifests,
            exclusions=[],
            thresholds=_thresholds(),
            prompt_template_sha256=DIGEST,
            style_guide_sha256=DIGEST,
            teacher_model_sha256=DIGEST,
            tokenizer_sha256=DIGEST,
            legacy_inventory={"schema_version": "test", "files": []},
            seal_permissions=False,
        )


def test_release_binds_uniform_manifest_provenance_and_excluded_meetings(
    tmp_path: Path,
) -> None:
    sft, manifests = _complete()
    manifests["eval"][0]["teacher_model_sha256"] = "a" * 64
    with pytest.raises(ReleaseValidationError, match="mix generation provenance"):
        assess_release_quality(
            sft_rows=sft,
            manifest_rows=manifests,
            exclusions=[],
            thresholds=_thresholds(),
        )

    sft, manifests = _complete()
    with pytest.raises(ReleaseValidationError, match="release provenance"):
        publish_release(
            release_root=tmp_path / "releases",
            release_id="wrong-provenance",
            sft_rows=sft,
            manifest_rows=manifests,
            exclusions=[],
            thresholds=_thresholds(),
            prompt_template_sha256=DIGEST,
            style_guide_sha256=DIGEST,
            teacher_model_sha256="a" * 64,
            tokenizer_sha256=DIGEST,
            legacy_inventory={"schema_version": "test", "files": []},
            seal_permissions=False,
        )

    # A meeting represented only by a terminal exclusion remains part of the
    # frozen split; acceptance thresholds, rather than membership, decide pass.
    excluded_sft = {split: list(rows) for split, rows in sft.items()}
    excluded_manifests = {split: list(rows) for split, rows in manifests.items()}
    excluded_sft["train"] = []
    excluded_manifests["train"] = []
    exclusion = {
        "sample_id": "2019-01-30::GDP Growth",
        "split": "train",
        "meeting_date": "2019-01-30",
        "atomic_topic": "GDP Growth",
        "reason_code": "candidate_rejected",
    }
    thresholds = replace(
        _thresholds(),
        train_acceptance_rate=0.0,
        per_topic_acceptance_rate=0.0,
    )
    quality, _, _ = assess_release_quality(
        sft_rows=excluded_sft,
        manifest_rows=excluded_manifests,
        exclusions=[exclusion],
        thresholds=thresholds,
    )
    assert quality["meeting_counts"]["train"] == 1
    assert quality["accepted_meeting_counts"]["train"] == 0
