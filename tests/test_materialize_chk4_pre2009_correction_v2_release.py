from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from jobs.retrain_v2 import (
    materialize_chk4_pre2009_correction_v2_release as correction,
)


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


def _make_writable(root: Path) -> None:
    root.chmod(0o700)
    for path in root.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)


def _copy_writable(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    _make_writable(destination)
    return destination


def _rebind_file_record(release: Path, relative: str) -> str:
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
    _write_json(manifest_path, manifest)
    return _sha256_file(manifest_path)


@pytest.fixture(scope="module")
def derived() -> dict[str, Any]:
    return correction._derive_all(correction.DEFAULT_PARENT)


@pytest.fixture(scope="module")
def published_release(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any]]:
    output = (
        tmp_path_factory.mktemp("chk4-correction-v2-publish") / correction.RELEASE_ID
    )
    verified = correction.materialize(correction.DEFAULT_PARENT, output)
    return output, verified


def test_real_derivation_has_frozen_counts_and_four_dimensional_exclusion(
    derived: dict[str, Any],
) -> None:
    audit = derived["audit"]
    assert len(derived["selected"]) == 33
    assert len(derived["schedule"]) == 48
    assert audit["repeat_histogram"] == {"1": 18, "2": 15}
    assert audit["burned_unique_rows"] == 14
    assert audit["burned_physical_rows"] == 16
    assert audit["unique_action_counts"] == correction.EXPECTED_UNIQUE_ACTIONS
    assert {
        name: receipt["source_ids"]["count"]
        for name, receipt in audit["partition_lineage_receipts"].items()
    } == {"selection": 86, "blind": 85, "smoke": 50, "retention": 69}
    assert (
        sum(
            receipt["source_ids"]["count"]
            for receipt in audit["partition_lineage_receipts"].values()
        )
        == 290
    )
    assert all(
        count == 0
        for receipt in audit["training_partition_overlap_counts"].values()
        for count in receipt.values()
    )
    assert all(
        count == 0
        for receipt in audit["fresh_partition_prior_overlap_counts"].values()
        for count in receipt.values()
    )


def test_partial_source_lineage_overlap_between_partitions_fails_closed(
    derived: dict[str, Any],
) -> None:
    rows = copy.deepcopy(derived["parent_rows"])
    by_id = {row["sample_id"]: row for row in rows}
    selection = by_id[correction.PARTITION_IDS["selection"][0]]
    blind = by_id[correction.PARTITION_IDS["blind"][0]]
    blind["source_ids"].append(selection["source_ids"][0])
    with pytest.raises(
        correction.CorrectionV2ReleaseError,
        match="partition lineage overlap.*source_ids",
    ):
        correction._derive_partitions(
            rows, set(correction.BURNED_SAMPLE_IDS), set(derived["v1_train_ids"])
        )


def test_fresh_partition_partial_source_overlap_with_prior_training_fails_closed(
    derived: dict[str, Any],
) -> None:
    rows = copy.deepcopy(derived["parent_rows"])
    by_id = {row["sample_id"]: row for row in rows}
    fresh = by_id[correction.PARTITION_IDS["selection"][0]]
    burned = by_id[correction.BURNED_SAMPLE_IDS[0]]
    fresh["source_ids"].append(burned["source_ids"][0])
    with pytest.raises(
        correction.CorrectionV2ReleaseError,
        match="fresh partition lineage overlaps",
    ):
        correction._derive_partitions(
            rows, set(correction.BURNED_SAMPLE_IDS), set(derived["v1_train_ids"])
        )


def test_training_partial_source_overlap_with_reserved_partition_fails_closed(
    derived: dict[str, Any],
) -> None:
    rows = copy.deepcopy(derived["parent_rows"])
    by_id = {row["sample_id"]: row for row in rows}
    burned = by_id[correction.BURNED_SAMPLE_IDS[0]]
    reserved = by_id[correction.PARTITION_IDS["selection"][0]]
    burned["source_ids"].append(reserved["source_ids"][0])
    excluded = {
        key: set().union(
            *(
                derived["partitions"]["lineage"][name][key]
                for name in derived["partitions"]["lineage"]
            )
        )
        for key in ("sample_id", "prompt_sha256", "meeting_date", "source_ids")
    }
    with pytest.raises(
        correction.CorrectionV2ReleaseError,
        match="training lineage overlaps reserved partitions: source_ids",
    ):
        correction._select_training_rows(
            rows,
            set(correction.BURNED_SAMPLE_IDS),
            excluded,
            set(derived["v1_train_ids"]),
        )


def test_real_temp_publish_is_sealed_and_deep_verified(
    published_release: tuple[Path, dict[str, Any]],
) -> None:
    release, verified = published_release
    assert verified["release_id"] == correction.RELEASE_ID
    assert verified["release_manifest_sha256"] == _sha256_file(
        release / "release_manifest.json"
    )
    assert not release.stat().st_mode & 0o222
    assert all(not path.stat().st_mode & 0o222 for path in release.rglob("*"))
    assert verified["materialization_runtime"]["python"] == {
        "executable": "/home/haobin_cui/.conda/envs/fomc_trainer/bin/python",
        "version": "3.10.9",
    }


def test_wrong_external_manifest_pin_fails_before_replay(
    published_release: tuple[Path, dict[str, Any]],
) -> None:
    release, _verified = published_release
    with pytest.raises(
        correction.CorrectionV2ReleaseError,
        match="external manifest pin mismatch",
    ):
        correction.verify_release(release, expected_manifest_sha256="0" * 64)


def test_runtime_binding_replays_release_and_exact_cp38_parent(
    published_release: tuple[Path, dict[str, Any]],
) -> None:
    release, verified = published_release
    binding = correction.verify_runtime_release(
        dataset_dir=release / "decision_sft",
        manifest_path=release / "release_manifest.json",
        expected_manifest_sha256=verified["release_manifest_sha256"],
        dataset_role=correction.DATASET_ROLE,
        model_path=correction.TRAINING_PARENT_MODEL,
    )
    assert binding["release_id"] == correction.RELEASE_ID
    assert binding["sampler_contract"]["schedule_path"] == str(
        release / "manifests/sampler_schedule.jsonl"
    )
    assert binding["sampler_contract"]["train_path"] == str(
        release / "decision_sft/train.jsonl"
    )
    assert binding["training_parent_model"]["sha256"] == (
        correction.TRAINING_PARENT_SHA256
    )


def test_teacher_response_substitution_fails_after_self_consistent_file_rehash(
    published_release: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    pristine, _verified = published_release
    release = _copy_writable(pristine, tmp_path / correction.RELEASE_ID)
    relative = "decision_sft/train.jsonl"
    rows = _read_jsonl(release / relative)
    replacement = next(
        row["response"] for row in rows if row["response"] != rows[0]["response"]
    )
    rows[0]["response"] = replacement
    _write_jsonl(release / relative, rows)
    rewritten_pin = _rebind_file_record(release, relative)
    with pytest.raises(
        correction.CorrectionV2ReleaseError, match="SFT train replay drift"
    ):
        correction.verify_release(
            release,
            expected_manifest_sha256=rewritten_pin,
            require_sealed=False,
        )


def test_handoff_semantic_tamper_fails_closed(
    published_release: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    pristine, verified = published_release
    release = _copy_writable(pristine, tmp_path / correction.RELEASE_ID)
    handoff_path = release / "handoff.json"
    handoff = _read_json(handoff_path)
    handoff["optimizer_steps"] = 7
    _write_json(handoff_path, handoff)
    with pytest.raises(
        correction.CorrectionV2ReleaseError,
        match="handoff semantic payload drift",
    ):
        correction.verify_release(
            release,
            expected_manifest_sha256=verified["release_manifest_sha256"],
            require_sealed=False,
        )


def test_v1_handoff_unsigned_payload_tamper_fails_closed(
    derived: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = correction._read_json

    def tampered(path: Path, *, label: str) -> dict[str, Any]:
        value = original(path, label=label)
        if path == correction.V1_RELEASE / "handoff.json":
            value["training_ready"] = False
        return value

    monkeypatch.setattr(correction, "_read_json", tampered)
    with pytest.raises(
        correction.CorrectionV2ReleaseError,
        match="correction-v1 handoff identity/readiness drift",
    ):
        correction._v1_training_ids(derived["parent_rows"])


def test_eexist_race_never_chmods_other_destination_and_cleans_sealed_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    output = tmp_path / correction.RELEASE_ID

    def fake_build(_parent: Path, staging: Path) -> dict[str, Any]:
        (staging / "release_manifest.json").write_text("{}\n", encoding="utf-8")
        return {}

    def fake_rename(_source: Path, destination: Path) -> None:
        destination.mkdir()
        destination.chmod(0o700)
        (destination / "belongs-to-other-publisher").write_text(
            "preserve\n", encoding="utf-8"
        )
        raise FileExistsError("simulated publish race")

    monkeypatch.setattr(correction, "build_release", fake_build)
    monkeypatch.setattr(correction, "verify_release", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(correction.legacy, "_rename_noreplace", fake_rename)

    with pytest.raises(FileExistsError, match="simulated publish race"):
        correction.materialize(parent, output)

    assert output.stat().st_mode & 0o777 == 0o700
    assert (output / "belongs-to-other-publisher").read_text(
        encoding="utf-8"
    ) == "preserve\n"
    assert not list(tmp_path.glob(f".{correction.RELEASE_ID}.*"))
