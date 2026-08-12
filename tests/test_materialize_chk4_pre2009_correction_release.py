from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from jobs.retrain_v2 import materialize_chk4_pre2009_correction_release as correction


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert all(isinstance(row, dict) for row in rows)
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _copy_release(source: Path, tmp_path: Path) -> Path:
    destination = tmp_path / correction.RELEASE_ID
    shutil.copytree(source, destination)
    return destination


def _rebind_file_record(
    release: Path,
    relative: str,
    *,
    rebind_evaluation: str | None = None,
) -> str:
    """Recompute visible hashes so deep replay, not the file table, rejects drift."""

    path = release / relative
    record: dict[str, Any] = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if path.suffix == ".jsonl":
        record["rows"] = len(path.read_text(encoding="utf-8").splitlines())

    manifest_path = release / "release_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["files"][relative] = record
    if rebind_evaluation is not None:
        inheritance = manifest["evaluation_inheritance"][rebind_evaluation]
        inheritance["sha256"] = record["sha256"]
        inheritance["rows"] = record["rows"]
    _write_json(manifest_path, manifest)
    return _sha256_file(manifest_path)


@pytest.fixture(scope="module")
def real_parent_child_release(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, str]:
    root = tmp_path_factory.mktemp("chk4-pre2009-correction") / correction.RELEASE_ID
    correction.build_release(correction.DEFAULT_PARENT, root)
    return root, _sha256_file(root / "release_manifest.json")


def test_real_parent_dry_run_does_not_create_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / correction.RELEASE_ID
    status = correction.main(
        [
            "--parent",
            str(correction.DEFAULT_PARENT),
            "--output",
            str(output),
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert status == 0
    assert result == {
        "status": "ready",
        "release_id": correction.RELEASE_ID,
        "dataset_role": correction.DATASET_ROLE,
        "selected_rows": 48,
        "schedule_rows": 48,
        "output": str(output.resolve()),
        "create_only": True,
    }
    assert not output.exists()


def test_real_parent_build_to_tmp_passes_deep_replay(
    real_parent_child_release: tuple[Path, str],
) -> None:
    release, manifest_sha = real_parent_child_release
    verified = correction.verify_release(
        release,
        expected_manifest_sha256=manifest_sha,
    )

    assert verified["release_id"] == correction.RELEASE_ID
    assert verified["release_manifest_sha256"] == manifest_sha
    assert verified["selection_contract"]["selected_unique_rows"] == 48
    assert verified["sampler_contract"]["optimizer_steps"] == 6
    assert verified["sampler_contract"]["repeat_histogram"] == {"1": 48}
    assert verified["unique_split_counts"] == {
        "train": 48,
        "validation": 13,
        "test": 13,
    }
    assert verified["input_contract"]["inherited_byte_for_byte"] is True
    assert verified["token_contract"]["tokenizer_bundle_sha256"] == (
        "c3100adca8f0f8f8f4c513a5361d41a43086ddd7aef049619d5c26cb3446a433"
    )
    assert verified["heldout_contract"]["holdout_scope"] == "correction_stage_only"
    assert verified["heldout_contract"]["historically_seen_by_parent"] is True
    assert (release / "handoff.json").is_file()


def test_deep_verify_requires_an_external_manifest_pin(
    real_parent_child_release: tuple[Path, str],
) -> None:
    release, _manifest_sha = real_parent_child_release
    with pytest.raises(
        correction.CorrectionReleaseError,
        match="requires an external manifest SHA-256 pin",
    ):
        correction.verify_release(release, expected_manifest_sha256="")


def test_preregistered_pre2009_holdouts_replay_hash_rank(
    real_parent_child_release: tuple[Path, str],
) -> None:
    release, _manifest_sha = real_parent_child_release
    manifest = _read_json(release / "release_manifest.json")
    rank = manifest["heldout_contract"]["preregistered_pre2009_hash_rank"]
    assert rank["algorithm"] == correction.HELDOUT_SELECTION_ALGORITHM
    assert rank["salt"] == "corrective-holdout-v1"
    assert rank["sample_ids_by_direction"] == {
        "hold": "dec-0bc683809351ff9da117027b",
        "hike": "dec-62f0a67d390dc9a866611dac",
    }


def test_selection_and_six_optimizer_windows_are_exactly_balanced(
    real_parent_child_release: tuple[Path, str],
) -> None:
    release, _manifest_sha = real_parent_child_release
    selected = _read_jsonl(release / "manifests/source_selection.jsonl")
    schedule = _read_jsonl(release / "manifests/sampler_schedule.jsonl")
    train = _read_jsonl(release / "decision_sft/train.jsonl")

    selected_ids = [str(row["sample_id"]) for row in selected]
    scheduled_ids = [str(row["source_sample_id"]) for row in schedule]
    assert len(selected) == len(schedule) == len(train) == 48
    assert len(set(selected_ids)) == len(set(scheduled_ids)) == 48
    assert set(selected_ids) == set(scheduled_ids)
    assert [row["schedule_index"] for row in schedule] == list(range(48))
    assert [row["source_sample_id"] for row in train] == scheduled_ids
    assert Counter(row["population_role"] for row in selected) == Counter(
        {"core": 18, "supplement": 30}
    )

    assert sorted({int(row["optimizer_step"]) for row in schedule}) == list(range(6))
    for optimizer_step in range(6):
        window = [
            row for row in schedule if int(row["optimizer_step"]) == optimizer_step
        ]
        assert len(window) == 8
        assert [int(row["microbatch_slot"]) for row in window] == list(range(8))
        assert Counter(row["direction"] for row in window) == Counter(
            {"hold": 3, "hike": 3, "cut": 2}
        )
        assert Counter(row["population_role"] for row in window) == Counter(
            {"core": 3, "supplement": 5}
        )
        assert len({row["source_sample_id"] for row in window}) == 8


def test_all_seven_heldout_ids_prompts_and_source_lineages_are_excluded(
    real_parent_child_release: tuple[Path, str],
) -> None:
    release, _manifest_sha = real_parent_child_release
    selected = _read_jsonl(release / "manifests/source_selection.jsonl")
    heldout = _read_jsonl(release / "manifests/heldout_probe.jsonl")

    heldout_ids = {str(row["sample_id"]) for row in heldout}
    heldout_prompts = {str(row["prompt_sha256"]) for row in heldout}
    heldout_sources = {
        str(source_id)
        for row in heldout
        for source_id in row.get("source_ids", [])
    }
    selected_ids = {str(row["sample_id"]) for row in selected}
    selected_prompts = {str(row["prompt_sha256"]) for row in selected}
    selected_sources = {
        str(source_id)
        for row in selected
        for source_id in row.get("source_ids", [])
    }

    assert len(heldout) == len(heldout_ids) == 7
    assert [row["sample_id"] for row in heldout] == list(correction.HELDOUT_SAMPLE_IDS)
    assert heldout_ids == set(correction.HELDOUT_SAMPLE_IDS)
    assert selected_ids.isdisjoint(heldout_ids)
    assert selected_prompts.isdisjoint(heldout_prompts)
    assert selected_sources.isdisjoint(heldout_sources)


def test_physical_parent_rows_and_duplicate_unique_ids_are_rejected() -> None:
    physical_rows = _read_jsonl(correction.DEFAULT_PARENT / "decision_sft/train.jsonl")
    unique_rows = correction._parent_source_rows(correction.DEFAULT_PARENT)

    assert len(physical_rows) == 312
    assert len(unique_rows) == len({row["sample_id"] for row in unique_rows}) == 211
    with pytest.raises(
        correction.CorrectionReleaseError,
        match="parent unique source IDs are not unique",
    ):
        correction.select_source_rows(physical_rows)

    duplicated = [dict(row) for row in unique_rows]
    duplicated.append(dict(unique_rows[0]))
    with pytest.raises(
        correction.CorrectionReleaseError,
        match="parent unique source IDs are not unique",
    ):
        correction.select_source_rows(duplicated)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_evaluation_byte_drift_is_rejected_after_self_consistent_rehash(
    split: str,
    real_parent_child_release: tuple[Path, str],
    tmp_path: Path,
) -> None:
    pristine, _manifest_sha = real_parent_child_release
    release = _copy_release(pristine, tmp_path)
    relative = f"decision_sft/{split}.jsonl"
    path = release / relative
    before = path.read_bytes()
    assert b"\n" in before
    path.write_bytes(before.replace(b"\n", b" \n", 1))
    rewritten_pin = _rebind_file_record(
        release,
        relative,
        rebind_evaluation=split,
    )

    with pytest.raises(
        correction.CorrectionReleaseError,
        match=rf"{split} inheritance drift",
    ):
        correction.verify_release(
            release,
            expected_manifest_sha256=rewritten_pin,
        )


def test_selection_tamper_is_rejected_by_semantic_replay_after_rehash(
    real_parent_child_release: tuple[Path, str],
    tmp_path: Path,
) -> None:
    pristine, _manifest_sha = real_parent_child_release
    release = _copy_release(pristine, tmp_path)
    relative = "manifests/source_selection.jsonl"
    path = release / relative
    rows = _read_jsonl(path)
    rows[0]["meeting_date"] = "1900-01-01"
    _write_jsonl(path, rows)
    rewritten_pin = _rebind_file_record(release, relative)

    with pytest.raises(
        correction.CorrectionReleaseError,
        match="source selection replay drift",
    ):
        correction.verify_release(
            release,
            expected_manifest_sha256=rewritten_pin,
        )


def test_previous_external_manifest_pin_rejects_manifest_rewrite(
    real_parent_child_release: tuple[Path, str],
    tmp_path: Path,
) -> None:
    pristine, pristine_pin = real_parent_child_release
    release = _copy_release(pristine, tmp_path)
    manifest_path = release / "release_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["created_at_utc"] = "2099-01-01T00:00:00Z"
    _write_json(manifest_path, manifest)
    assert _sha256_file(manifest_path) != pristine_pin

    with pytest.raises(
        correction.CorrectionReleaseError,
        match="correction release manifest hash drift",
    ):
        correction.verify_release(
            release,
            expected_manifest_sha256=pristine_pin,
        )


def test_handoff_tamper_is_rejected_by_unsigned_binding(
    real_parent_child_release: tuple[Path, str],
    tmp_path: Path,
) -> None:
    pristine, pristine_pin = real_parent_child_release
    release = _copy_release(pristine, tmp_path)
    handoff_path = release / "handoff.json"
    handoff = _read_json(handoff_path)
    handoff["optimizer_steps"] = 7
    _write_json(handoff_path, handoff)

    with pytest.raises(
        correction.CorrectionReleaseError,
        match="handoff unsigned payload binding drift",
    ):
        correction.verify_release(
            release,
            expected_manifest_sha256=pristine_pin,
        )


def test_create_only_materialize_rejects_occupied_output_without_mutation(
    tmp_path: Path,
) -> None:
    output = tmp_path / correction.RELEASE_ID
    output.mkdir()
    sentinel = output / "belongs-to-caller.txt"
    sentinel.write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(correction.CorrectionReleaseError, match="output exists"):
        correction.materialize(correction.DEFAULT_PARENT, output)

    assert sentinel.read_text(encoding="utf-8") == "preserve me\n"
    assert sorted(path.name for path in output.iterdir()) == [sentinel.name]
