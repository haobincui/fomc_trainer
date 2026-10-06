from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from jobs.eval import score_chk3_core8_loo_vllm_k10 as subject
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


def _sample_map() -> dict[tuple[int, int], tuple[str, str, str]]:
    return {
        (meeting, variant): (
            f"meeting-{meeting:03d}",
            f"sample-{meeting:03d}-{variant:02d}",
            f"reference-{meeting:03d}",
        )
        for meeting in range(subject.EXPECTED_MEETINGS)
        for variant in range(subject.EXPECTED_VARIANTS)
    }


def _old_rows() -> list[dict]:
    rows = []
    for meeting in range(subject.EXPECTED_MEETINGS):
        for variant in range(subject.EXPECTED_VARIANTS):
            arm, topic = subject._expected_variant(variant)
            for replicate in range(subject.OLD_REPLICATES):
                position = (
                    (meeting * subject.EXPECTED_VARIANTS + variant)
                    * subject.OLD_REPLICATES
                    + replicate
                )
                row = {
                    "schema_version": subject.k5.SCORE_ROW_SCHEMA,
                    "meeting_id": f"meeting-{meeting:03d}",
                    "meeting_rank": meeting,
                    "variant_rank": variant,
                    "arm": arm,
                    "intervention_topic": topic,
                    "replicate_id": replicate,
                    "replicate_seed": subject.OLD_REPLICATE_SEEDS[replicate],
                    "row_seed": meeting * 100 + replicate,
                    "sample_id": f"sample-{meeting:03d}-{variant:02d}",
                    "reference_minutes_sha256": f"reference-{meeting:03d}",
                    "raw_semantic_scores": {
                        "mpnet_cosine": 0.5,
                        "bertscore_f1": 0.6,
                        "bertscore_precision": 0.61,
                        "bertscore_recall": 0.59,
                    },
                    "generation_diagnostics": {
                        "metrics": {},
                        "preregistered_core_valid": True,
                        "preregistered_core_failures": [],
                    },
                    "primary_loo_policy": {
                        "included": True,
                        "gate_filter_applied": False,
                        "gate_zero_penalty_applied": False,
                    },
                    "tuple_key": {
                        "absolute_case_index": position,
                        "meeting_id": f"meeting-{meeting:03d}",
                        "meeting_rank": meeting,
                        "variant_rank": variant,
                        "arm": arm,
                        "intervention_topic": topic,
                        "replicate_id": replicate,
                        "replicate_seed": subject.OLD_REPLICATE_SEEDS[replicate],
                        "row_seed": meeting * 100 + replicate,
                    },
                    "completion_sha256": f"completion-old-{position}",
                    "answer_sha256": f"answer-old-{position}",
                    "generated_token_ids_sha256": f"tokens-old-{position}",
                    "finish_reason": "eos",
                    "input_truncated": False,
                    "generated_tokens": 10,
                    "empty_answer": False,
                    "full_token_4gram_repetition": 0.0,
                    "tail_token_4gram_repetition": 0.0,
                }
                rows.append(row)
    return rows


def _increment_generation_rows() -> list[dict]:
    profile = subject._require_increment_profile()
    rows = []
    for meeting in range(subject.EXPECTED_MEETINGS):
        for variant in range(subject.EXPECTED_VARIANTS):
            arm, topic = subject._expected_variant(variant)
            for local in range(subject.INCREMENT_REPLICATES):
                position = (
                    (meeting * subject.EXPECTED_VARIANTS + variant)
                    * subject.INCREMENT_REPLICATES
                    + local
                )
                global_id = subject.GLOBAL_REPLICATE_OFFSET + local
                rows.append(
                    {
                        "schema_version": profile.ROW_SCHEMA,
                        "evaluation_id": profile.EVALUATION_ID,
                        "model_id": "chk3",
                        "absolute_case_index": position,
                        "meeting_id": f"meeting-{meeting:03d}",
                        "meeting_rank": meeting,
                        "variant_rank": variant,
                        "arm": arm,
                        "intervention_topic": topic,
                        "replicate_id": local,
                        "local_replicate_id": local,
                        "global_replicate_id": global_id,
                        "replicate_seed": subject.INCREMENT_REPLICATE_SEEDS[local],
                        "paired_block_id": meeting * subject.INCREMENT_REPLICATES
                        + local,
                        "row_seed": meeting * 100 + global_id,
                        "paired_seed_key": f"meeting-{meeting:03d}-full",
                        "variant_sample_id": f"sample-{meeting:03d}-{variant:02d}",
                        "reference_minutes_sha256": f"reference-{meeting:03d}",
                        "answer": "answer",
                        "reference_minutes": "reference",
                        "input_truncated": False,
                        "finish_reason": "eos",
                    }
                )
    return rows


def _increment_scored_rows() -> list[dict]:
    rows = []
    for source in _increment_generation_rows():
        global_id = source["global_replicate_id"]
        key = subject._combined_tuple_key(source, global_id)
        rows.append(
            {
                "schema_version": subject.SCORE_ROW_SCHEMA,
                "source_release": "increment_generation_k6_to_k10",
                "combined_score_row": key["absolute_case_index"] + 1,
                "tuple_key": key,
                "sample_id": source["variant_sample_id"],
                "meeting_id": source["meeting_id"],
                "meeting_rank": source["meeting_rank"],
                "variant_rank": source["variant_rank"],
                "arm": source["arm"],
                "intervention_topic": source["intervention_topic"],
                "replicate_id": global_id,
                "replicate_seed": source["replicate_seed"],
                "row_seed": source["row_seed"],
                "reference_minutes_sha256": source["reference_minutes_sha256"],
            }
        )
    return rows


def _matrices(value: float = 0.02) -> dict[tuple[str, str, str], np.ndarray]:
    return {
        (topic, arm, metric): np.full(
            (subject.EXPECTED_MEETINGS, subject.EXPECTED_REPLICATES),
            value,
            dtype=np.float64,
        )
        for topic in subject.TOPICS
        for arm in subject.ARMS
        for metric in subject.METRICS
    }


def _meeting_ids() -> list[str]:
    return [f"meeting-{index:03d}" for index in range(subject.EXPECTED_MEETINGS)]


def test_incremental_k10_frozen_contract_and_launcher() -> None:
    assert subject.EXPECTED_INCREMENT_ROWS == 10_880
    assert subject.EXPECTED_ROWS == 21_760
    assert subject.EXPECTED_PAIRED_BLOCKS == 1_280
    assert subject.GLOBAL_REPLICATE_IDS == (5, 6, 7, 8, 9)
    assert subject.REPLICATE_SEEDS == (
        20260811,
        21260811,
        22260811,
        23260811,
        24260811,
        25260811,
        26260811,
        27260811,
        28260811,
        29260811,
    )
    launcher = (
        subject.ROOT / "run/score_chk3_core8_loo_vllm_k10_dual_gpu.sh"
    ).read_text(encoding="utf-8")
    assert "generation_formal_increment_k6_k10_n2176_v1/chk3/manifest.json" in launcher
    assert "score_incremental_reuse_k5_raw_semantic_b10000_v1" in launcher
    assert "old_k5_rows_rescored" not in launcher


def test_k10_source_bundle_binds_native_probe_and_score_launcher() -> None:
    sources = subject._source_bundle()
    assert set(subject.OLD_K5_IMPLEMENTATION_COMPATIBILITY_ROLES).issubset(sources)
    assert sources["native_probe_gate_engine"]["path"] == str(
        Path(subject.k5.semantic.native_eval.native_probe.__file__).resolve()
    )
    assert sources["score_launcher"]["path"] == str(
        subject.SCORE_LAUNCHER.resolve()
    )
    assert sources["score_launcher"]["sha256"] == subject._sha256_file(
        subject.SCORE_LAUNCHER
    )


def test_old_k5_source_bundle_is_fully_revalidated_and_cross_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_manifest = subject._read_json(subject.DEFAULT_OLD_SCORE_MANIFEST)
    current = subject._source_bundle()
    original = subject.k5._assert_source_bundle
    calls: list[object] = []

    def spy(value: object) -> dict:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(subject.k5, "_assert_source_bundle", spy)
    validated = subject._validate_old_k5_source_compatibility(
        old_manifest, current
    )
    assert calls == [old_manifest["sources"]]
    assert set(validated) == set(subject.k5.SOURCE_ROLES)

    # Isolate the explicit cross-release check from the full K5 validator and
    # prove native-probe drift is independently fatal.
    monkeypatch.setattr(subject.k5, "_assert_source_bundle", lambda value: {})
    drifted = copy.deepcopy(old_manifest)
    drifted["sources"]["native_probe_gate_engine"]["sha256"] = "0" * 64
    with pytest.raises(
        subject.LooK10ScoreError, match="native_probe_gate_engine"
    ):
        subject._validate_old_k5_source_compatibility(drifted, current)


def test_merged_increment_manifest_uses_nested_full_indexing_contract() -> None:
    profile = subject._require_increment_profile()
    manifest = {
        "status": "complete",
        "evaluation_id": profile.EVALUATION_ID,
        "model_id": "chk3",
        "coverage": {"cases": subject.EXPECTED_INCREMENT_ROWS},
        "generation_contract": {
            "replicate_indexing": profile.preparation.replicate_indexing_contract()
        },
    }
    subject.validate_increment_manifest_contract(manifest)
    broken = copy.deepcopy(manifest)
    broken["generation_contract"]["replicate_indexing"][
        "global_replicate_offset"
    ] = 0
    with pytest.raises(subject.LooK10ScoreError, match="indexing contract drift"):
        subject.validate_increment_manifest_contract(broken)
    top_level_only = copy.deepcopy(manifest)
    top_level_only["replicate_indexing"] = top_level_only[
        "generation_contract"
    ].pop("replicate_indexing")
    with pytest.raises(subject.LooK10ScoreError, match="indexing contract drift"):
        subject.validate_increment_manifest_contract(top_level_only)


def test_increment_rows_validate_local_to_global_offset_and_seeds() -> None:
    rows = _increment_generation_rows()
    coverage = subject.validate_increment_rows(rows, old_sample_map=_sample_map())
    assert coverage["rows"] == 10_880
    assert coverage["paired_blocks"] == 640
    assert coverage["local_replicate_ids"] == [0, 1, 2, 3, 4]
    assert coverage["global_replicate_ids"] == [5, 6, 7, 8, 9]
    broken = copy.deepcopy(rows)
    broken[0]["global_replicate_id"] = 0
    with pytest.raises(subject.LooK10ScoreError, match="mapping drift"):
        subject.validate_increment_rows(broken, old_sample_map=_sample_map())


def test_combined_projection_is_exact_128_by_17_by_10() -> None:
    old = _old_rows()
    subject.validate_old_score_rows(old)
    combined = subject.combine_scored_rows(old, _increment_scored_rows())
    assert len(combined) == 21_760
    assert [row["tuple_key"]["absolute_case_index"] for row in combined] == list(
        range(21_760)
    )
    assert {row["replicate_id"] for row in combined} == set(range(10))
    first_variant = combined[:10]
    assert [row["replicate_id"] for row in first_variant] == list(range(10))
    assert [row["replicate_seed"] for row in first_variant] == list(
        subject.REPLICATE_SEEDS
    )


def test_shared_k10_hierarchical_plan_has_all_prefixes() -> None:
    first = subject.make_bootstrap_plan(meeting_ids=_meeting_ids(), draws=20, seed=19)
    same = subject.make_bootstrap_plan(meeting_ids=_meeting_ids(), draws=20, seed=19)
    assert first.sha256 == same.sha256
    assert first.meeting_indices.shape == (20, 128)
    assert first.replicate_indices.shape == (20, 128, 10)
    assert list(first.prefix_replicate_indices) == list(range(1, 11))
    for prefix, indexes in first.prefix_replicate_indices.items():
        assert indexes.shape == (20, 128, prefix)
        assert int(indexes.min()) >= 0
        assert int(indexes.max()) < prefix


def test_constant_k10_panel_is_adequate_and_sealed() -> None:
    computation = subject.compute_bootstrap(
        matrices=_matrices(0.02), meeting_ids=_meeting_ids(), draws=50, seed=23
    )
    validate_manifest_integrity(computation.results)
    cell = computation.results["cells"][subject.TOPICS[0]][subject.ARMS[0]][
        subject.METRICS[0]
    ]
    assert math.isclose(cell["point_estimate_mean_delta"], 0.02)
    assert math.isclose(cell["hierarchical_bootstrap"]["ci_lower"], 0.02)
    assert math.isclose(cell["hierarchical_bootstrap"]["ci_upper"], 0.02)
    assert list(
        cell["k_adequacy"]["cumulative_fresh_stochastic_prefixes"]
    ) == [f"k{value}" for value in range(1, 11)]
    assert computation.results["coverage"]["paired_blocks"] == 1_280
    assert computation.results["k_adequacy"]["increase_k_recommended"] is False


def test_k10_minus_k9_drift_triggers_frozen_gate() -> None:
    matrices = _matrices(0.0)
    target = (subject.TOPICS[0], subject.ARMS[0], subject.METRICS[0])
    matrices[target][:, 9] = 0.10
    result = subject.compute_bootstrap(
        matrices=matrices, meeting_ids=_meeting_ids(), draws=50, seed=29
    ).results
    diagnostic = result["cells"][target[0]][target[1]][target[2]]["k_adequacy"]
    assert math.isclose(diagnostic["k10_minus_k9_point_drift"], 0.01)
    assert "k10_minus_k9_point_drift_above_threshold" in diagnostic[
        "increase_k_reasons"
    ]


def test_create_only_json_publication_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    subject._write_new_json(path, {"value": 1})
    with pytest.raises(subject.k5.LooK5ScoreError, match="refusing to overwrite"):
        subject._write_new_json(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}
