from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from jobs.eval import bootstrap_chk3_core8_loo_full as subject
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


def _synthetic_rows(
    *, meetings: int = 3, topics: tuple[str, ...] = ("A", "B")
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for meeting_index in range(meetings):
        meeting_id = f"meeting-{meeting_index:02d}"
        full_mpnet = 0.60 + 0.01 * meeting_index
        full_bert = 0.70 + 0.01 * meeting_index
        common = {
            "schema_version": subject.SOURCE_ROW_SCHEMA,
            "meeting_id": meeting_id,
            "reference_minutes_sha256": "a" * 64,
        }
        rows.append(
            {
                **common,
                "sample_id": f"{meeting_id}:full",
                "arm": "full",
                "intervention_topic": None,
                "mpnet_cosine_raw": full_mpnet,
                "bertscore_f1_raw": full_bert,
            }
        )
        for topic_index, topic in enumerate(topics):
            for arm in subject.ARMS:
                if arm == "exact_deletion":
                    mpnet_delta = 0.10 * (topic_index + 1)
                    bert_delta = 0.05 * (topic_index + 1)
                else:
                    mpnet_delta = -0.02 * (topic_index + 1)
                    bert_delta = 0.03 * (topic_index + 1)
                rows.append(
                    {
                        **common,
                        "sample_id": f"{meeting_id}:{arm}:{topic}",
                        "arm": arm,
                        "intervention_topic": topic,
                        "mpnet_cosine_raw": full_mpnet - mpnet_delta,
                        "bertscore_f1_raw": full_bert - bert_delta,
                    }
                )
    return rows


def _source_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    computation = subject.compute_bootstrap(
        rows,
        draws=5,
        seed=7,
        expected_meetings=3,
        expected_topics=2,
    )
    topic_deltas: dict[str, object] = {}
    for topic in computation.topics:
        topic_deltas[topic] = {}
        for arm in subject.ARMS:
            topic_deltas[topic][arm] = {
                "mpnet_delta_raw_mean": computation.results["cells"][topic][arm][
                    "mpnet_cosine"
                ]["point_estimate_mean_delta"],
                "bertscore_f1_delta_raw_mean": computation.results["cells"][topic][
                    arm
                ]["bertscore_f1"]["point_estimate_mean_delta"],
            }
    return {"topic_deltas": topic_deltas}


def _write_source_bundle(tmp_path: Path) -> tuple[Path, str]:
    rows = _synthetic_rows()
    row_path = tmp_path / "row_scores.jsonl"
    row_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    generation_path = tmp_path / "generation_manifest.json"
    generation = seal_manifest(
        {
            "schema_version": "synthetic-generation-v1",
            "status": "complete",
            "generation_contract": {
                "mode": "greedy",
                "do_sample": False,
                "batch_size": 1,
            },
        }
    )
    generation_path.write_text(
        json.dumps(generation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    score_path = tmp_path / "score_manifest.json"
    score = seal_manifest(
        {
            "schema_version": subject.SOURCE_SCORE_SCHEMA,
            "status": "complete",
            "evaluation_id": subject.EVALUATION_ID,
            "run_manifest": subject._file_binding(generation_path, sealed=True),
            "artifacts": {
                "row_scores": subject._file_binding(row_path, rows=len(rows))
            },
            "summary": _source_summary(rows),
        }
    )
    score_path.write_text(
        json.dumps(score, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return score_path, subject._sha256_file(score_path)


def test_formal_contract_is_b10000_with_date_seed() -> None:
    assert subject.BOOTSTRAP_DRAWS == 10_000
    assert subject.BOOTSTRAP_SEED == 20_260_815
    assert len(subject.ARMS) * len(subject.METRICS) * subject.EXPECTED_TOPICS == 32


def test_shared_index_plan_is_deterministic_and_applied_to_every_cell() -> None:
    rows = _synthetic_rows()
    first = subject.compute_bootstrap(
        rows,
        draws=100,
        seed=11,
        expected_meetings=3,
        expected_topics=2,
    )
    second = subject.compute_bootstrap(
        rows,
        draws=100,
        seed=11,
        expected_meetings=3,
        expected_topics=2,
    )
    changed = subject.compute_bootstrap(
        rows,
        draws=100,
        seed=12,
        expected_meetings=3,
        expected_topics=2,
    )
    assert first.index_plan_sha256 == second.index_plan_sha256
    assert np.array_equal(first.meeting_indices, second.meeting_indices)
    assert not np.array_equal(first.meeting_indices, changed.meeting_indices)
    for key, vector in first.delta_vectors.items():
        expected = vector[first.meeting_indices].mean(axis=1)
        np.testing.assert_array_equal(first.draw_means[key], expected)


def test_constant_delta_has_degenerate_ci_and_sign_share_one() -> None:
    result = subject.compute_bootstrap(
        _synthetic_rows(),
        draws=100,
        seed=11,
        expected_meetings=3,
        expected_topics=2,
    ).results["cells"]["A"]["exact_deletion"]["mpnet_cosine"]
    assert math.isclose(result["point_estimate_mean_delta"], 0.1)
    assert math.isclose(result["ci_lower"], 0.1)
    assert math.isclose(result["ci_upper"], 0.1)
    assert result["positive_meeting_sign_share"] == 1.0
    assert result["bootstrap_draw_share_above_zero"] == 1.0


def test_k_adequacy_fails_closed_on_within_prompt_uncertainty() -> None:
    diagnostic = subject.compute_bootstrap(
        _synthetic_rows(),
        draws=10,
        seed=11,
        expected_meetings=3,
        expected_topics=2,
    ).results["k_adequacy"]
    assert diagnostic["generation_replicates_per_meeting_arm"] == 1
    assert diagnostic["replicate_resampling_performed"] is False
    assert diagnostic["within_prompt_decoding_variance_estimable"] is False
    assert diagnostic["meeting_panel_resampling_estimable"] is True


def test_formal_index_matrix_has_frozen_raw_hash() -> None:
    _, _, metadata = subject._index_plan(
        meeting_ids=tuple(f"meeting-{index:03d}" for index in range(128)),
        draws=10_000,
        seed=20_260_815,
    )
    assert metadata["raw_indices_sha256"] == (
        "b76daca538bfd37192f9f80bb54731122623631b8e76bd12e3c376f1ecf003c8"
    )


def test_end_to_end_bundle_is_sealed_and_refuses_overwrite(tmp_path: Path) -> None:
    score_path, score_sha = _write_source_bundle(tmp_path)
    output = tmp_path / "bootstrap"
    manifest = subject.bootstrap_score_bundle(
        score_manifest_path=score_path,
        score_manifest_sha256=score_sha,
        output_dir=output,
        draws=50,
        seed=13,
        expected_meetings=3,
        expected_topics=2,
    )
    assert validate_manifest_integrity(manifest) == manifest["integrity"][
        "payload_sha256"
    ]
    assert manifest["execution"]["generation_performed"] is False
    assert manifest["execution"]["semantic_scoring_performed"] is False
    assert manifest["artifacts"]["bootstrap_draws"]["rows"] == 50
    assert validate_manifest_integrity(
        json.loads((output / "bootstrap_results.json").read_text(encoding="utf-8"))
    )
    with pytest.raises(subject.Core8LooBootstrapError, match="overwrite"):
        subject.bootstrap_score_bundle(
            score_manifest_path=score_path,
            score_manifest_sha256=score_sha,
            output_dir=output,
            draws=50,
            seed=13,
            expected_meetings=3,
            expected_topics=2,
        )
