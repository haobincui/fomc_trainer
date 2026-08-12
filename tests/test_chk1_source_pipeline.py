from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2.chk1 import source_pipeline
from jobs.retrain_v2.chk1.source_pipeline import (
    SourcePlanError,
    create_source_plan,
    load_meetings_by_split,
    materialize_source_plan,
    validate_source_handoff,
)


def _dates(start: int, count: int) -> list[str]:
    # Canonical, strictly increasing synthetic dates without calendar math.
    return [
        f"{2000 + (start + index) // 12:04d}-{(start + index) % 12 + 1:02d}-01"
        for index in range(count)
    ]


def _population(path: Path, population_id: str, dates: list[str], split: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "loo-population-v1",
                "population_id": population_id,
                "phase": "test",
                "split_label": split,
                "meeting_dates": dates,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _reseal_handoff(handoff: dict[str, object]) -> None:
    payload = {key: value for key, value in handoff.items() if key != "payload_sha256"}
    handoff["payload_sha256"] = source_pipeline.sha256_text(
        source_pipeline._canonical_json(payload)
    )


def test_102_13_13_roster_uses_supported_populations_without_resplitting(
    tmp_path: Path,
) -> None:
    train = _dates(0, 102)
    evaluation = _dates(120, 13)
    test = _dates(144, 13)
    eval_path = tmp_path / "eval.json"
    test_path = tmp_path / "test.json"
    _population(eval_path, "pilot_eval_13", evaluation, "eval")
    _population(test_path, "formal_test_13", test, "test")

    path = create_source_plan(
        meetings_by_split={"train": train, "eval": evaluation, "test": test},
        output_dir=tmp_path / "plan",
        existing_populations={"eval": eval_path, "test": test_path},
    )
    plan = json.loads(path.read_text(encoding="utf-8"))

    assert plan["meeting_counts"] == {"train": 102, "eval": 13, "test": 13}
    train_populations = [
        item for item in plan["populations"] if item["split"] == "train"
    ]
    assert [len(item["meeting_dates"]) for item in train_populations] == [
        13,
        13,
        13,
        13,
        13,
        13,
        13,
        11,
    ]
    assert [item["expected_vintage_count"] for item in plan["snapshot_batches"]] == [
        26,
        26,
        26,
        13,
        11,
    ]
    assert [
        date for item in train_populations for date in item["meeting_dates"]
    ] == train
    assert all(Path(item["path"]).is_file() for item in train_populations)


def test_source_plan_rejects_overlap_and_unrepresentable_split(tmp_path: Path) -> None:
    train = _dates(0, 13)
    evaluation = [train[0], *_dates(30, 12)]
    evaluation.sort()
    with pytest.raises(SourcePlanError, match="overlap"):
        create_source_plan(
            meetings_by_split={
                "train": train,
                "eval": evaluation,
                "test": _dates(50, 13),
            },
            output_dir=tmp_path / "overlap",
        )

    with pytest.raises(SourcePlanError, match="cannot be partitioned"):
        create_source_plan(
            meetings_by_split={
                "train": _dates(0, 12),
                "eval": _dates(20, 13),
                "test": _dates(40, 13),
            },
            output_dir=tmp_path / "unsupported",
        )


def test_load_meetings_rejects_cross_split_membership(tmp_path: Path) -> None:
    root = tmp_path / "student"
    root.mkdir()
    for split, dates in {
        "train": ["2020-01-01"],
        "eval": ["2020-01-01"],
        "test": ["2020-03-01"],
    }.items():
        (root / f"{split}.jsonl").write_text(
            json.dumps({"split": split, "meeting_date": dates[0]}) + "\n",
            encoding="utf-8",
        )
    with pytest.raises(SourcePlanError, match="both train and eval"):
        load_meetings_by_split(root)


def test_materialize_replays_each_ledger_with_all_bound_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = create_source_plan(
        meetings_by_split={
            "train": _dates(0, 13),
            "eval": _dates(30, 13),
            "test": _dates(60, 13),
        },
        output_dir=tmp_path / "plan",
    )
    registry = tmp_path / "registry.json"
    roster = tmp_path / "roster.json"
    registry.write_text("{}\n", encoding="utf-8")
    roster.write_text("{}\n", encoding="utf-8")

    def fake_fetch_source_snapshots(*, output_dir: Path, **_: object) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "snapshot_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": source_pipeline.DENSE_SNAPSHOT_SCHEMA_VERSION,
                    "status": "complete",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"status": "complete"}

    def fake_build_ledger(*, output_dir: Path, **_: object) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "ledger_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": source_pipeline.DENSE_LEDGER_MANIFEST_SCHEMA_VERSION,
                    "status": "complete",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    validation_calls: list[dict[str, object]] = []

    def fake_validate_ledger(**kwargs: object) -> dict[str, object]:
        validation_calls.append(kwargs)
        return {
            "manifest_payload_sha256": "a" * 64,
            "row_count": 338,
        }

    monkeypatch.setattr(
        source_pipeline, "fetch_source_snapshots", fake_fetch_source_snapshots
    )
    monkeypatch.setattr(
        source_pipeline, "build_loo_indicator_ledger", fake_build_ledger
    )
    monkeypatch.setattr(
        source_pipeline, "validate_loo_indicator_ledger", fake_validate_ledger
    )

    handoff_path = materialize_source_plan(
        plan_path=plan_path,
        registry_path=registry,
        roster_path=roster,
        output_dir=tmp_path / "source",
    )

    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert len(validation_calls) == 3
    assert len(handoff["ledgers"]) == 3
    assert {record["coverage_mode"] for record in handoff["ledgers"]} == {"dense"}
    for call in validation_calls:
        assert set(call) == {
            "ledger_manifest_file",
            "registry_file",
            "snapshot_manifest_file",
            "population_file",
            "roster_file",
        }
        assert call["registry_file"] == registry.resolve()
        assert call["roster_file"] == roster.resolve()

    for record in handoff["ledgers"]:
        record.pop("coverage_mode")
    _reseal_handoff(handoff)
    handoff_path.write_text(json.dumps(handoff) + "\n", encoding="utf-8")

    resumed = materialize_source_plan(
        plan_path=plan_path,
        registry_path=registry,
        roster_path=roster,
        output_dir=tmp_path / "source",
        resume=True,
    )
    assert resumed == handoff_path
    assert len(validation_calls) == 6

    tampered = json.loads(handoff_path.read_text(encoding="utf-8"))
    tampered["registry_sha256"] = "0" * 64
    handoff_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(SourcePlanError, match="payload hash mismatch"):
        materialize_source_plan(
            plan_path=plan_path,
            registry_path=registry,
            roster_path=roster,
            output_dir=tmp_path / "source",
            resume=True,
        )


def test_sparse_train_dispatches_and_binds_release_exclusions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = create_source_plan(
        meetings_by_split={
            "train": _dates(0, 13),
            "eval": _dates(30, 13),
            "test": _dates(60, 13),
        },
        output_dir=tmp_path / "plan",
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    populations = {item["population_id"]: item for item in plan["populations"]}
    registry = tmp_path / "registry.json"
    roster = tmp_path / "roster.json"
    registry.write_text("{}\n", encoding="utf-8")
    roster.write_text("{}\n", encoding="utf-8")

    dense_fetches: list[dict[str, object]] = []
    sparse_acquisitions: list[dict[str, object]] = []
    sparse_transports: list[object] = []
    dense_validations: list[dict[str, object]] = []
    sparse_snapshot_validations: list[dict[str, object]] = []
    sparse_ledger_validations: list[dict[str, object]] = []

    def fake_dense_fetch(
        *,
        output_dir: Path,
        population_paths: list[Path],
        expected_vintage_count: int,
        **_: object,
    ) -> dict[str, str]:
        dense_fetches.append(
            {
                "population_paths": population_paths,
                "expected_vintage_count": expected_vintage_count,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "snapshot_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": source_pipeline.DENSE_SNAPSHOT_SCHEMA_VERSION,
                    "status": "complete",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"status": "complete"}

    def fake_sparse_acquire(
        *,
        output_dir: Path,
        population_id: str,
        split: str,
        meeting_dates: list[str],
        **_: object,
    ) -> Path:
        sparse_transports.append(_.get("http_get"))
        sparse_acquisitions.append(
            {
                "population_id": population_id,
                "split": split,
                "meeting_dates": meeting_dates,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "snapshot_manifest.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": (
                        source_pipeline.SPARSE_SNAPSHOT_MANIFEST_SCHEMA_VERSION
                    ),
                    "status": "complete",
                    "population_id": population_id,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def fake_dense_build(*, output_dir: Path, **_: object) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "ledger_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": source_pipeline.DENSE_LEDGER_MANIFEST_SCHEMA_VERSION,
                    "status": "complete",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def fake_sparse_build(*, output_dir: Path, **_: object) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sample_exclusions.jsonl").write_text(
            '{"reason_code":"sealed-unusable"}\n',
            encoding="utf-8",
        )
        path = output_dir / "ledger_manifest.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": (
                        source_pipeline.SPARSE_LEDGER_MANIFEST_SCHEMA_VERSION
                    ),
                    "status": "complete",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def fake_dense_validate(**kwargs: object) -> dict[str, object]:
        dense_validations.append(kwargs)
        return {"manifest_payload_sha256": "a" * 64, "row_count": 338}

    def fake_sparse_snapshot_validate(
        _manifest: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        sparse_snapshot_validations.append(kwargs)
        return {
            "status": "valid",
            "population_id": kwargs["expected_population_id"],
            "manifest_payload_sha256": "b" * 64,
        }

    def fake_sparse_ledger_validate(
        manifest: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        sparse_ledger_validations.append({"manifest": manifest, **kwargs})
        return {
            "status": "valid",
            "population_id": manifest.parent.name,
            "manifest_payload_sha256": "c" * 64,
            "ready_sample_count": 337,
            "excluded_sample_count": 1,
            "expected_sample_count": 338,
        }

    monkeypatch.setattr(source_pipeline, "fetch_source_snapshots", fake_dense_fetch)
    monkeypatch.setattr(
        source_pipeline,
        "acquire_sparse_source_snapshots",
        fake_sparse_acquire,
    )
    monkeypatch.setattr(
        source_pipeline,
        "build_loo_indicator_ledger",
        fake_dense_build,
    )
    monkeypatch.setattr(source_pipeline, "build_sparse_loo_ledger", fake_sparse_build)
    monkeypatch.setattr(
        source_pipeline,
        "validate_loo_indicator_ledger",
        fake_dense_validate,
    )
    monkeypatch.setattr(
        source_pipeline,
        "validate_sparse_snapshot_manifest",
        fake_sparse_snapshot_validate,
    )
    monkeypatch.setattr(
        source_pipeline,
        "validate_sparse_loo_ledger",
        fake_sparse_ledger_validate,
    )

    output_dir = tmp_path / "source"
    sparse_transport = lambda _url, _timeout: None
    handoff_path = materialize_source_plan(
        plan_path=plan_path,
        registry_path=registry,
        roster_path=roster,
        output_dir=output_dir,
        sparse_train=True,
        sparse_http_get=sparse_transport,
    )
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    records = {record["population_id"]: record for record in handoff["ledgers"]}
    train_id = next(
        population_id
        for population_id, population in populations.items()
        if population["split"] == "train"
    )
    train_record = records[train_id]

    assert sparse_acquisitions == [
        {
            "population_id": train_id,
            "split": "train",
            "meeting_dates": populations[train_id]["meeting_dates"],
        }
    ]
    assert sparse_transports == [sparse_transport]
    assert len(dense_fetches) == 2
    assert all(call["expected_vintage_count"] == 13 for call in dense_fetches)
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["split_label"] != "train"
        for call in dense_fetches
        for path in call["population_paths"]
    )
    assert train_record["coverage_mode"] == "sparse"
    assert train_record["row_count"] == 337
    assert train_record["excluded_sample_count"] == 1
    assert train_record["expected_sample_count"] == 338
    assert train_record["snapshot_payload_sha256"] == "b" * 64
    assert train_record["ledger_payload_sha256"] == "c" * 64
    assert train_record["sample_exclusions"]["row_count"] == 1
    assert len(train_record["sample_exclusions"]["sha256"]) == 64
    assert {
        record["coverage_mode"]
        for population_id, record in records.items()
        if population_id != train_id
    } == {"dense"}
    assert len(dense_validations) == 2
    assert len(sparse_snapshot_validations) == 1
    assert len(sparse_ledger_validations) == 1

    validated = validate_source_handoff(
        handoff_path=handoff_path,
        plan_path=plan_path,
        registry_path=registry,
        roster_path=roster,
    )
    assert validated == handoff
    assert len(dense_validations) == 4
    assert len(sparse_snapshot_validations) == 2
    assert len(sparse_ledger_validations) == 2

    with pytest.raises(SourcePlanError, match="requested run"):
        materialize_source_plan(
            plan_path=plan_path,
            registry_path=registry,
            roster_path=roster,
            output_dir=output_dir,
            resume=True,
        )
    assert (
        materialize_source_plan(
            plan_path=plan_path,
            registry_path=registry,
            roster_path=roster,
            output_dir=output_dir,
            sparse_train=True,
            resume=True,
        )
        == handoff_path
    )

    original_handoff = handoff_path.read_text(encoding="utf-8")
    incomplete = json.loads(original_handoff)
    incomplete["ledgers"] = incomplete["ledgers"][:-1]
    _reseal_handoff(incomplete)
    handoff_path.write_text(json.dumps(incomplete) + "\n", encoding="utf-8")
    with pytest.raises(SourcePlanError, match="population coverage mismatch"):
        validate_source_handoff(
            handoff_path=handoff_path,
            plan_path=plan_path,
            registry_path=registry,
            roster_path=roster,
        )

    handoff_path.write_text(original_handoff, encoding="utf-8")
    exclusion_path = Path(train_record["sample_exclusions"]["path"])
    exclusion_path.write_text(
        exclusion_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(SourcePlanError, match="sample-exclusion binding changed"):
        validate_source_handoff(
            handoff_path=handoff_path,
            plan_path=plan_path,
            registry_path=registry,
            roster_path=roster,
        )


@pytest.mark.parametrize(
    ("sparse_train", "reuse_split", "manifest_schema"),
    [
        (True, "train", source_pipeline.DENSE_SNAPSHOT_SCHEMA_VERSION),
        (True, "eval", source_pipeline.SPARSE_SNAPSHOT_MANIFEST_SCHEMA_VERSION),
        (False, "train", source_pipeline.SPARSE_SNAPSHOT_MANIFEST_SCHEMA_VERSION),
    ],
)
def test_reused_snapshot_schema_cannot_cross_coverage_modes(
    tmp_path: Path,
    sparse_train: bool,
    reuse_split: str,
    manifest_schema: str,
) -> None:
    plan_path = create_source_plan(
        meetings_by_split={
            "train": _dates(0, 13),
            "eval": _dates(30, 13),
            "test": _dates(60, 13),
        },
        output_dir=tmp_path / "plan",
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    population_id = next(
        item["population_id"]
        for item in plan["populations"]
        if item["split"] == reuse_split
    )
    manifest = tmp_path / "reused-snapshot.json"
    manifest.write_text(
        json.dumps({"schema_version": manifest_schema, "status": "complete"}) + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    roster = tmp_path / "roster.json"
    registry.write_text("{}\n", encoding="utf-8")
    roster.write_text("{}\n", encoding="utf-8")

    with pytest.raises(SourcePlanError, match="schema mismatch"):
        materialize_source_plan(
            plan_path=plan_path,
            registry_path=registry,
            roster_path=roster,
            output_dir=tmp_path / "source",
            reused_snapshots={population_id: manifest},
            sparse_train=sparse_train,
        )


def test_cli_sparse_train_switch_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_materialize(**kwargs: object) -> Path:
        calls.append(kwargs)
        return tmp_path / "source_handoff.json"

    monkeypatch.setattr(source_pipeline, "materialize_source_plan", fake_materialize)
    result = source_pipeline.main(
        [
            "materialize",
            "--plan",
            str(tmp_path / "plan.json"),
            "--registry",
            str(tmp_path / "registry.json"),
            "--roster",
            str(tmp_path / "roster.json"),
            "--output-dir",
            str(tmp_path / "source"),
            "--sparse-train",
        ]
    )
    assert result == 0
    assert calls[0]["sparse_train"] is True

    default_args = source_pipeline.build_parser().parse_args(
        [
            "materialize",
            "--plan",
            "plan.json",
            "--registry",
            "registry.json",
            "--roster",
            "roster.json",
            "--output-dir",
            "source",
        ]
    )
    assert default_args.sparse_train is False
