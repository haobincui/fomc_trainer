import hashlib
import json
import stat
from pathlib import Path

import pytest

from jobs.retrain_v2 import build_totalsl_unit_overlay as overlay


def _canonical_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _evidence(
    series_id: str,
    evidence_id: str,
    *,
    units: str = "Percent",
    value: str = "1.25",
):
    return {
        "availability_basis": "sealed_fixture",
        "evidence_id": evidence_id,
        "fact_kind": "latest",
        "metric": f"Metric {series_id}",
        "observation_date": "2026-01-01",
        "series_id": series_id,
        "source_sha256": "a" * 64,
        "units": units,
        "value": value,
    }


def _row(sample_id: str, evidence, *, duplicate_prompt_data: bool = False):
    card = {
        "atomic_topic": "Bank Credit to Private Sector",
        "evidence": evidence,
        "schema_version": "fixture-v1",
    }
    provided_data = _canonical_bytes(card).decode("utf-8")
    prompt = "Analyze the supplied point-in-time evidence.\n\n" + provided_data
    if duplicate_prompt_data:
        prompt += "\n\n" + provided_data
    return {
        "meeting_date": "2026-01-28",
        "prompt": prompt,
        "provided_data": provided_data,
        "sample_id": sample_id,
    }


def _fixture_rows(*, wrong_units: bool = False, duplicate_prompt_data: bool = False):
    totalsl_units = "Thousands of Dollars" if wrong_units else overlay.SOURCE_UNITS
    return {
        "train": [
            _row(
                "train-0",
                [
                    _evidence("TOTALSL", "train-0-a", units=totalsl_units),
                    _evidence("OTHER", "train-0-b", units=overlay.SOURCE_UNITS),
                    _evidence("TOTALSL", "train-0-c", units=totalsl_units),
                ],
                duplicate_prompt_data=duplicate_prompt_data,
            ),
            _row("train-1", [_evidence("TOTALSLAR", "train-1-a")]),
            _row(
                "train-2",
                [_evidence("TOTALSL", "train-2-a", units=totalsl_units)],
            ),
        ],
        "eval": [
            _row(
                "eval-0",
                [_evidence("TOTALSL", "eval-0-a", units=totalsl_units)],
            ),
            _row("eval-1", [_evidence("OTHER", "eval-1-a")]),
        ],
        "test": [
            _row(
                "test-0",
                [_evidence("TOTALSL", "test-0-a", units=totalsl_units)],
            )
        ],
    }


def _seal_source(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)


def _bound_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    wrong_units: bool = False,
    duplicate_prompt_data: bool = False,
):
    parent = tmp_path / "dataset/processed/retrain_v2"
    source_release = tmp_path / overlay.SOURCE_RELEASE_RELATIVE
    source_dataset = tmp_path / overlay.SOURCE_DATASET_RELATIVE
    source_dataset.mkdir(parents=True)
    rows = _fixture_rows(
        wrong_units=wrong_units,
        duplicate_prompt_data=duplicate_prompt_data,
    )

    split_contract = {}
    split_records = {}
    expected_affected = {"train": 2, "eval": 1, "test": 1}
    expected_entries = {"train": 3, "eval": 1, "test": 1}
    for split in overlay.SPLITS:
        payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows[split])
        (source_dataset / f"{split}.jsonl").write_bytes(payload)
        manifest_key = "validation" if split == "eval" else split
        split_contract[split] = {
            "manifest_key": manifest_key,
            "rows": len(rows[split]),
            "affected_rows": expected_affected[split],
            "evidence_entries": expected_entries[split],
            "sha256": _sha(payload),
        }
        split_records[manifest_key] = {
            "path": f"analysis_grpo/{split}.jsonl",
            "sha256": _sha(payload),
        }

    dataset_artifact_sha = overlay._directory_artifact_sha256(source_dataset)
    manifest = {
        "datasets": {
            "analysis_grpo": {
                "artifact_sha256": dataset_artifact_sha,
                "path": "analysis_grpo",
                "split_files": split_records,
            }
        },
        "generated_at_utc": "2026-08-04T11:04:39Z",
        "release_id": overlay.SOURCE_RELEASE_ID,
        "release_type": "analysis_base",
        "schema_version": 1,
    }
    manifest_payload = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    (source_release / "base_release_manifest.json").write_bytes(manifest_payload)
    _seal_source(source_release)

    monkeypatch.setattr(overlay, "SOURCE_SPLIT_CONTRACT", split_contract)
    monkeypatch.setattr(
        overlay,
        "EXPECTED_SOURCE_MANIFEST_SHA256",
        _sha(manifest_payload),
    )
    monkeypatch.setattr(
        overlay,
        "EXPECTED_SOURCE_DATASET_ARTIFACT_SHA256",
        dataset_artifact_sha,
    )
    monkeypatch.setattr(overlay, "EXPECTED_TOTAL_ROWS", 6)
    monkeypatch.setattr(overlay, "EXPECTED_AFFECTED_ROWS", 4)
    monkeypatch.setattr(overlay, "EXPECTED_EVIDENCE_ENTRIES", 5)
    monkeypatch.setattr(overlay, "EXPECTED_UNCHANGED_ROWS", 2)
    return {
        "parent": parent,
        "rows": rows,
        "source_dataset": source_dataset,
        "source_release": source_release,
    }


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_bound_production_parent_has_exact_totalsl_population() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = overlay._load_source_snapshot(repo_root)
    transformed = overlay._transform_snapshot(source)

    assert {
        split: (
            len(transformed[split].rows),
            transformed[split].affected_rows,
            transformed[split].evidence_entries,
        )
        for split in overlay.SPLITS
    } == {
        "train": (493, 30, 46),
        "eval": (199, 12, 12),
        "test": (190, 13, 13),
    }
    assert {split: transformed[split].sha256 for split in overlay.SPLITS} == {
        "train": "6d7a16249bd81bbcdcdcfd8897d5d87f1d30aa0b31a9891d2e628f4de304a815",
        "eval": "bf7ca73a2f8a648965d1d97ff8d6a12610fa280337e41db77ebe912f7bbeda9c",
        "test": "0a48000bb7757e3842a3f22be0eae69725b2e24e0388a8f276e0477418f0fa6c",
    }


def test_builder_publishes_only_bound_unit_changes_with_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _bound_repo(tmp_path, monkeypatch)
    source_payloads = {
        split: (fixture["source_dataset"] / f"{split}.jsonl").read_bytes()
        for split in overlay.SPLITS
    }

    destination = overlay.build_totalsl_unit_overlay(
        tmp_path,
        generated_at_utc="2026-08-10T12:30:00Z",
    )

    assert destination == tmp_path / overlay.DESTINATION_RELATIVE
    assert {
        path.name for path in destination.iterdir()
    } == {
        "train.jsonl",
        "eval.jsonl",
        "test.jsonl",
        overlay.MANIFEST_FILENAME,
        overlay.ATTESTATION_FILENAME,
    }
    for split in overlay.SPLITS:
        source_lines = source_payloads[split].splitlines(keepends=True)
        output_payload = (destination / f"{split}.jsonl").read_bytes()
        output_lines = output_payload.splitlines(keepends=True)
        assert len(output_payload) == len(source_payloads[split])
        for source_line, output_line in zip(source_lines, output_lines, strict=True):
            source_row = json.loads(source_line)
            output_row = json.loads(output_line)
            source_card = json.loads(source_row["provided_data"])
            output_card = json.loads(output_row["provided_data"])
            target_indexes = [
                index
                for index, item in enumerate(source_card["evidence"])
                if item["series_id"] == overlay.SERIES_ID
            ]
            if target_indexes:
                assert output_line != source_line
                for index in target_indexes:
                    assert output_card["evidence"][index]["units"] == overlay.TARGET_UNITS
                assert output_row["prompt"].count(output_row["provided_data"]) == 1
            else:
                assert output_line == source_line

    train_card = json.loads(_read_jsonl(destination / "train.jsonl")[0]["provided_data"])
    assert train_card["evidence"][1]["series_id"] == "OTHER"
    assert train_card["evidence"][1]["units"] == overlay.SOURCE_UNITS

    manifest = json.loads((destination / overlay.MANIFEST_FILENAME).read_text())
    assert manifest["counts"] == {
        "affected_rows": 4,
        "evidence_entries_changed": 5,
        "rows": 6,
        "unchanged_rows": 2,
    }
    assert manifest["transformation"] == {
        "field": "evidence[*].units",
        "from": overlay.SOURCE_UNITS,
        "prompt_projection": "replace the unique embedded provided_data copy",
        "series_id": "TOTALSL",
        "to": overlay.TARGET_UNITS,
    }
    attestation = json.loads(
        (destination / overlay.ATTESTATION_FILENAME).read_text()
    )
    assert len(attestation["changed_rows"]) == 4
    assert sum(
        row["totalsl_evidence_entries"] for row in attestation["changed_rows"]
    ) == 5
    unsigned = dict(attestation)
    claimed = unsigned.pop("attestation_payload_sha256")
    assert claimed == overlay._canonical_sha256(unsigned)
    assert overlay.validate_totalsl_unit_overlay(tmp_path)["status"] == "valid"

    writable = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    assert all(
        path.stat().st_mode & writable == 0
        for path in (destination, *destination.rglob("*"))
    )
    assert not list(fixture["parent"].glob(f".{overlay.OVERLAY_ID}.build-*"))
    with pytest.raises(overlay.TotalslOverlayError, match="already exists"):
        overlay.build_totalsl_unit_overlay(tmp_path)


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"wrong_units": True}, "source units are not bound millions"),
        ({"duplicate_prompt_data": True}, "exactly once"),
    ],
)
def test_builder_fails_closed_on_source_semantic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_kwargs,
    message: str,
) -> None:
    fixture = _bound_repo(tmp_path, monkeypatch, **fixture_kwargs)

    with pytest.raises(overlay.TotalslOverlayError, match=message):
        overlay.build_totalsl_unit_overlay(tmp_path)

    assert not (tmp_path / overlay.DESTINATION_RELATIVE).exists()
    assert not list(fixture["parent"].glob(f".{overlay.OVERLAY_ID}.build-*"))


def test_existing_destination_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bound_repo(tmp_path, monkeypatch)
    destination = tmp_path / overlay.DESTINATION_RELATIVE
    destination.mkdir()
    marker = destination / "owned.txt"
    marker.write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(overlay.TotalslOverlayError, match="already exists"):
        overlay.build_totalsl_unit_overlay(tmp_path)

    assert marker.read_text(encoding="utf-8") == "preserve me\n"


def test_stale_staging_fails_closed_without_deleting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _bound_repo(tmp_path, monkeypatch)
    stale = fixture["parent"] / f".{overlay.OVERLAY_ID}.build-interrupted"
    stale.mkdir()
    marker = stale / "partial.json"
    marker.write_text("{}\n", encoding="utf-8")

    with pytest.raises(overlay.TotalslOverlayError, match="stale .* staging"):
        overlay.build_totalsl_unit_overlay(tmp_path)

    assert marker.is_file()
    assert not (tmp_path / overlay.DESTINATION_RELATIVE).exists()


def test_atomic_publication_failure_cleans_only_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _bound_repo(tmp_path, monkeypatch)

    def fail_rename(_source, _destination):
        raise overlay.TotalslOverlayError("injected atomic failure")

    monkeypatch.setattr(overlay, "_rename_noreplace", fail_rename)
    with pytest.raises(overlay.TotalslOverlayError, match="injected atomic failure"):
        overlay.build_totalsl_unit_overlay(tmp_path)

    assert not (tmp_path / overlay.DESTINATION_RELATIVE).exists()
    assert not list(fixture["parent"].glob(f".{overlay.OVERLAY_ID}.build-*"))


def test_source_drift_during_staging_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _bound_repo(tmp_path, monkeypatch)
    real_write = overlay._write_fsynced

    def mutate_source_after_attestation(path: Path, payload: bytes):
        real_write(path, payload)
        if path.name == overlay.ATTESTATION_FILENAME:
            source_train = fixture["source_dataset"] / "train.jsonl"
            source_train.chmod(0o600)
            source_train.write_bytes(source_train.read_bytes() + b"\n")
            source_train.chmod(0o444)

    monkeypatch.setattr(overlay, "_write_fsynced", mutate_source_after_attestation)
    with pytest.raises(overlay.TotalslOverlayError, match="changed during overlay build"):
        overlay.build_totalsl_unit_overlay(tmp_path)

    assert not (tmp_path / overlay.DESTINATION_RELATIVE).exists()
    assert not list(fixture["parent"].glob(f".{overlay.OVERLAY_ID}.build-*"))


def test_validator_rejects_output_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bound_repo(tmp_path, monkeypatch)
    destination = overlay.build_totalsl_unit_overlay(
        tmp_path,
        generated_at_utc="2026-08-10T12:30:00Z",
    )
    train_path = destination / "train.jsonl"
    train_path.chmod(0o600)
    train_path.write_bytes(
        train_path.read_bytes().replace(b"Billions of Dollars", b"Trillions of Dollars", 1)
    )
    train_path.chmod(0o444)

    with pytest.raises(overlay.TotalslOverlayError, match="output train bytes mismatch"):
        overlay.validate_totalsl_unit_overlay(tmp_path)
