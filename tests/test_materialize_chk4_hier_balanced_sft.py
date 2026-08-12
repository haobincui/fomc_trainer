from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from jobs.retrain_v2 import materialize_chk4_hier_balanced_sft as balanced
from open_r1.data_loader import load_train_eval_datasets


def _parent_unique_rows() -> list[dict[str, object]]:
    return balanced._read_jsonl(
        balanced.DEFAULT_PARENT / "manifests/unique/train.jsonl",
        label="hier-balanced test parent unique train",
    )


def test_schedule_is_deterministic_complete_and_window_balanced() -> None:
    unique_rows = _parent_unique_rows()
    first, first_ids = balanced.build_schedule(unique_rows)
    second, second_ids = balanced.build_schedule(copy.deepcopy(unique_rows))

    assert first == second
    assert first_ids == second_ids
    assert len(first) == balanced.TRAIN_ROWS == 192
    assert Counter(row["direction"] for row in first) == balanced.DIRECTION_COUNTS
    assert len(set(first_ids)) == balanced.PARENT_UNIQUE_ROWS

    unique_by_id = {row["sample_id"]: row for row in unique_rows}
    exposure = Counter(first_ids)
    for direction, expected in balanced.EXPECTED_REPEAT_HISTOGRAMS.items():
        observed = Counter(
            count
            for sample_id, count in exposure.items()
            if unique_by_id[sample_id]["direction"] == direction
        )
        assert dict(observed) == expected

    for step in range(balanced.OPTIMIZER_STEPS):
        window = first[
            step * balanced.EFFECTIVE_BATCH_SIZE : (step + 1)
            * balanced.EFFECTIVE_BATCH_SIZE
        ]
        assert Counter(row["direction"] for row in window) == balanced.PER_WINDOW_COUNTS
        assert len({row["source_sample_id"] for row in window}) == 8
        assert [row["schedule_index"] for row in window] == list(
            range(step * 8, (step + 1) * 8)
        )


def test_schedule_repeat_indices_are_zero_based_and_parent_hash_bound() -> None:
    unique_rows = _parent_unique_rows()
    unique_by_id = {row["sample_id"]: row for row in unique_rows}
    schedule, _ids = balanced.build_schedule(unique_rows)
    occurrences: dict[str, list[int]] = {}
    for row in schedule:
        sample_id = row["source_sample_id"]
        occurrences.setdefault(sample_id, []).append(row["source_repeat_index"])
        source = unique_by_id[sample_id]
        assert row["prompt_sha256"] == source["prompt_sha256"]
        assert row["response_sha256"] == source["response_sha256"]
        assert row["gold_sha256"] == source["gold_sha256"]
    for row in schedule:
        indices = occurrences[row["source_sample_id"]]
        assert indices == list(range(row["source_repeat_total"]))


def test_schedule_validator_rejects_source_or_window_tamper() -> None:
    unique_rows = _parent_unique_rows()
    schedule, _ids = balanced.build_schedule(unique_rows)

    source_tamper = copy.deepcopy(schedule)
    source_tamper[0]["prompt_sha256"] = "0" * 64
    with pytest.raises(balanced.HierBalancedReleaseError, match="parent prompt_sha256"):
        balanced._validate_schedule(source_tamper, unique_rows=unique_rows)

    window_tamper = copy.deepcopy(schedule)
    window_tamper[0], window_tamper[1] = window_tamper[1], window_tamper[0]
    for index, row in enumerate(window_tamper):
        row["schedule_index"] = index
    with pytest.raises(balanced.HierBalancedReleaseError):
        balanced._validate_schedule(window_tamper, unique_rows=unique_rows)


def test_real_parent_publish_verify_loader_and_create_only(tmp_path: Path) -> None:
    destination = tmp_path / balanced.RELEASE_ID
    result = balanced.publish(destination=destination)
    assert result["status"] == "published"
    assert result["release_id"] == balanced.RELEASE_ID

    verified = balanced.verify_release(
        destination,
        expected_manifest_sha256=result["release_manifest_sha256"],
        tokenizer=balanced.DEFAULT_TOKENIZER,
    )
    sampler = verified["sampler_contract"]
    assert sampler["schedule_rows"] == 192
    assert sampler["optimizer_steps"] == 24
    assert sampler["direction_counts"] == balanced.DIRECTION_COUNTS
    assert sampler["per_optimizer_window"] == balanced.PER_WINDOW_COUNTS
    assert sampler["order_is_authoritative"] is True
    assert sampler["secondary_shuffle_forbidden"] is True

    for split in ("validation", "test"):
        assert (destination / "decision_sft" / f"{split}.jsonl").read_bytes() == (
            balanced.DEFAULT_PARENT / "decision_sft" / f"{split}.jsonl"
        ).read_bytes()

    audit = json.loads(
        (destination / "audits/data_quality.json").read_text(encoding="utf-8")
    )
    assert audit["lineage_rows_checked"] == 192
    assert audit["new_prompt_rows"] == audit["new_response_rows"] == 0
    assert audit["unique_train_rows_covered"] == 102
    assert all(
        0.20 <= share <= 0.55
        for share in audit["train_supervised_completion_token_share"].values()
    )
    assert all(
        stats["prompt_overflow_rows"] == 0
        and stats["completion_overflow_rows"] == 0
        and stats["sft_overflow_rows"] == 0
        for stats in audit["split_token_stats"].values()
    )

    split_files = verified["verified_split_files"]
    loaded = load_train_eval_datasets(
        destination / "decision_sft",
        split_files=split_files,
    )
    assert len(loaded["train"]) == 192
    assert len(loaded["validation"]) == 13
    assert {"prompt", "response", "schedule_index", "training_row_id"}.issubset(
        loaded["train"].column_names
    )

    manifest_before = (destination / "release_manifest.json").read_bytes()
    with pytest.raises(
        balanced.HierBalancedReleaseError,
        match="immutable destination already exists",
    ):
        balanced.publish(destination=destination)
    assert (destination / "release_manifest.json").read_bytes() == manifest_before

    balanced._unseal_tree(destination)


def test_published_validator_rejects_file_tamper(tmp_path: Path) -> None:
    destination = tmp_path / balanced.RELEASE_ID
    result = balanced.publish(destination=destination)
    balanced._unseal_tree(destination)
    schedule_path = destination / "manifests/sampler_schedule.jsonl"
    rows = balanced._read_jsonl(schedule_path, label="tamper schedule")
    rows[0]["prompt_sha256"] = "0" * 64
    schedule_path.write_text(
        "".join(balanced._canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    balanced._seal_tree(destination)
    with pytest.raises(
        balanced.HierBalancedReleaseError,
        match="hash drift: manifests/sampler_schedule.jsonl",
    ):
        balanced.verify_release(
            destination,
            expected_manifest_sha256=result["release_manifest_sha256"],
        )
    balanced._unseal_tree(destination)
