from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from jobs.eval import evaluate_van_den_hauwe_2013_reduced as vdh


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _previous_month(month: str) -> str:
    first = date.fromisoformat(f"{month}-01")
    return (first - timedelta(days=1)).strftime("%Y-%m")


def _h19_targets(config: dict[str, Any]) -> dict[str, str]:
    payload = json.loads(vdh.resolve_path(config["historical_h19"]).read_text())
    return {
        row["meeting_start_date"][:7]: row["direction"] for row in payload["meetings"]
    }


def _source_release(tmp_path: Path, config: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    source = tmp_path / "source"
    source.mkdir()
    artifacts = {
        "dfedtar.csv": "observation_date,DFEDTAR\n",
        "tb6ms.csv": "observation_date,TB6MS\n",
        "fedfunds.csv": "observation_date,FEDFUNDS\n",
        "indpro_vintages.csv": "observation_date,INDPRO_vintage\n",
        "cpiaucsl_vintages.csv": "observation_date,CPIAUCSL_vintage\n",
    }
    for name, content in artifacts.items():
        (source / name).write_text(content, encoding="utf-8")

    months = vdh._month_sequence("1990-01", "2008-06")
    h19 = _h19_targets(config)
    pre = [month for month in months if month <= "2000-12"]
    oos = [month for month in months if month >= "2001-01"]
    observed_pre = {month for month in h19 if month <= "2000-12"}
    observed_oos = {month for month in h19 if month >= "2001-01"}
    observed_pre.update(month for month in pre if month not in observed_pre)
    observed_pre = set(sorted(observed_pre)[:95]) | {
        month for month in h19 if month <= "2000-12"
    }
    while len(observed_pre) > 95:
        removable = max(month for month in observed_pre if month not in h19)
        observed_pre.remove(removable)
    observed_oos.update(month for month in oos if month not in observed_oos)
    observed_oos = set(sorted(observed_oos)[:62]) | {
        month for month in h19 if month >= "2001-01"
    }
    while len(observed_oos) > 62:
        removable = max(month for month in observed_oos if month not in h19)
        observed_oos.remove(removable)
    observed = observed_pre | observed_oos
    assert len(observed_pre) == 95
    assert len(observed_oos) == 62
    assert len(observed) == 157

    source_paths = {
        "DFEDTAR": "dfedtar.csv",
        "TB6MS": "tb6ms.csv",
        "FEDFUNDS": "fedfunds.csv",
        "INDPRO": "indpro_vintages.csv",
        "CPIAUCSL": "cpiaucsl_vintages.csv",
    }
    source_hashes = {
        key: _sha256(source / path) for key, path in source_paths.items()
    }
    six_tff_hash = vdh.sha256_bytes(
        vdh.canonical_json(
            {"TB6MS": source_hashes["TB6MS"], "FEDFUNDS": source_hashes["FEDFUNDS"]}
        ).encode("utf-8")
    )
    remaining_directions: list[str] = ["cut"] * 36 + ["hold"] * 76 + ["hike"] * 26
    non_h19_observed = sorted(observed - set(h19))
    generated_direction = dict(zip(non_h19_observed, remaining_directions, strict=True))
    rows: list[dict[str, Any]] = []
    target_rate = 5.0
    for index, month in enumerate(months):
        first = date.fromisoformat(f"{month}-01")
        origin = (first - timedelta(days=1)).isoformat()
        decision_observed = month in observed
        direction = h19.get(month, generated_direction.get(month))
        previous_target_rate = target_rate
        if direction == "cut":
            target_rate -= 0.25
        elif direction == "hike":
            target_rate += 0.25
        if first.month == 12:
            next_first = date(first.year + 1, 1, 1)
        else:
            next_first = date(first.year, first.month + 1, 1)
        target_observation_text = (next_first - timedelta(days=1)).isoformat()
        predictors: dict[str, dict[str, Any]] = {}
        for predictor_index, spec in enumerate(config["predictors"]):
            paper_id = spec["paper_id"]
            value: dict[str, Any] = {
                "value": float(index + predictor_index + 1) / 10.0,
                "transform": spec["transform"],
                "averaging_months": spec["m"],
                "observation_period": _previous_month(month),
                "observation_dates": [origin],
                "availability_date": origin,
            }
            if paper_id == "6TFF":
                value.update(
                    {
                        "real_time_vintage": False,
                        "vintage_date": None,
                        "source_policy": "historical_market_nonrevised",
                        "series_ids": ["TB6MS", "FEDFUNDS"],
                        "source_sha256": six_tff_hash,
                    }
                )
            else:
                series_id = "INDPRO" if paper_id == "IP" else "CPIAUCSL"
                value.update(
                    {
                        "real_time_vintage": True,
                        "vintage_date": origin,
                        "source_policy": "alfred_end_of_prior_month_vintage",
                        "series_ids": [series_id],
                        "source_sha256": source_hashes[series_id],
                    }
                )
            predictors[paper_id] = value
        row = {
            "schema_version": "van-den-hauwe-2013-reduced-month-v1",
            "month": month,
            "information_cutoff": origin,
            "target_rate_end_month_pct": target_rate,
            "target_rate_observation_date": target_observation_text,
            "previous_target_rate_pct": previous_target_rate,
            "previous_target_rate_observation_date": origin,
            "decision_month": decision_observed,
            "scheduled_meeting_month": decision_observed,
            "unscheduled_only_decision_month": False,
            "direction": direction if decision_observed else None,
            "predictors": predictors,
            "target_source_sha256": source_hashes["DFEDTAR"],
        }
        row["row_sha256"] = vdh.sha256_bytes(vdh.canonical_json(row).encode("utf-8"))
        rows.append(row)
    monthly = source / "monthly_source.jsonl"
    _write_jsonl(monthly, rows)
    manifest = {
        "schema_version": vdh.SOURCE_SCHEMA,
        "config": str(vdh.DEFAULT_CONFIG.relative_to(vdh.ROOT)),
        "config_sha256": _sha256(vdh.DEFAULT_CONFIG),
        "sample": {
            "start_month": "1990-01",
            "end_month": "2008-06",
            "months": 222,
            "decision_months": 157,
            "scheduled_meeting_months": 157,
            "unscheduled_only_decision_months": 0,
            "direction_counts": {"cut": 40, "hold": 86, "hike": 31},
        },
        "predictors": [
            {
                "paper_id": "6TFF",
                "series_ids": ["TB6MS", "FEDFUNDS"],
                "transform": "av",
                "m": 1,
                "source_policy": "historical_market_nonrevised",
            },
            {
                "paper_id": "IP",
                "series_ids": ["INDPRO"],
                "transform": "gr",
                "m": 1,
                "source_policy": "alfred_end_of_prior_month_vintage",
            },
            {
                "paper_id": "INF",
                "series_ids": ["CPIAUCSL"],
                "transform": "gr",
                "m": 1,
                "source_policy": "alfred_end_of_prior_month_vintage",
            },
        ],
        "current_vintage_fallback_allowed": False,
        "market_history_caveat": "TB6MS and FEDFUNDS are treated as non-revised histories.",
        "files": [
            {
                "provider": "derived",
                "series_id": "monthly_source",
                "path": "monthly_source.jsonl",
                "sha256": _sha256(monthly),
            },
            *[
                {
                    "provider": "alfred" if series_id in {"INDPRO", "CPIAUCSL"} else "fred",
                    "series_id": series_id,
                    "path": source_paths[series_id],
                    "sha256": source_hashes[series_id],
                    **(
                        {"vintage_dates": [vdh._previous_month_end(value) for value in months]}
                        if series_id in {"INDPRO", "CPIAUCSL"}
                        else {}
                    ),
                }
                for series_id in ("DFEDTAR", "TB6MS", "FEDFUNDS", "INDPRO", "CPIAUCSL")
            ],
        ],
        "monthly_source_sha256": _sha256(monthly),
    }
    _write_json(source / "manifest.json", manifest)
    return source, rows


def _rewrite_monthly_source(source: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["row_sha256"] = vdh.sha256_bytes(
            vdh.canonical_json(
                {key: value for key, value in row.items() if key != "row_sha256"}
            ).encode("utf-8")
        )
    monthly = source / "monthly_source.jsonl"
    _write_jsonl(monthly, rows)
    manifest = json.loads((source / "manifest.json").read_text())
    for item in manifest["files"]:
        if item["series_id"] == "monthly_source":
            item["sha256"] = _sha256(monthly)
    manifest["monthly_source_sha256"] = _sha256(monthly)
    _write_json(source / "manifest.json", manifest)


def _config() -> dict[str, Any]:
    return vdh.load_config(vdh.DEFAULT_CONFIG)


def test_frozen_config_uses_exact_reduced_predictor_ids_and_time_contracts() -> None:
    config = _config()

    assert [row["paper_id"] for row in config["predictors"]] == ["6TFF", "IP", "INF"]
    assert [row["transform"] for row in config["predictors"]] == ["av", "gr", "gr"]
    assert [row["m"] for row in config["predictors"]] == [1, 1, 1]
    assert [row["source_policy"] for row in config["predictors"]] == [
        "historical_market_nonrevised",
        "real_time_vintage",
        "real_time_vintage",
    ]
    assert tuple(config["contracts"]) == vdh.CONTRACTS
    assert config["sample"]["expected_months"] == 222
    assert config["sample"]["expected_observed_decision_months"] == 157
    assert config["model"]["mcmc"]["burn_in"] == 50_000
    assert config["model"]["mcmc"]["draws"] == 100_000
    assert vdh.MIN_RECURSIVE_PRE_UPDATE_ESS == 100.0


def test_prepare_builds_222_months_preserves_missing_decisions_and_resumes(
    tmp_path: Path,
) -> None:
    config = _config()
    source, _ = _source_release(tmp_path, config)
    paths = vdh.runtime_paths(config, tmp_path / "run")

    manifest = vdh.run_prepare(
        config_path=vdh.DEFAULT_CONFIG,
        config=config,
        source_root=source.resolve(),
        paths=paths,
        resume=False,
    )
    panel = vdh.read_jsonl(paths.monthly_panel)

    assert manifest["monthly_panel_rows"] == len(panel) == 222
    assert sum(row["decision_observed"] for row in panel) == 157
    assert sum(row["target_direction"] is None for row in panel) == 65
    assert sum(row["matched_h19"] is not None for row in panel) == 19
    assert sum(row["paper_recursive_oos"] for row in panel) == 62
    assert panel[0]["predictors"]["6TFF"]["source_policy"] == "historical_market_nonrevised"
    assert panel[0]["predictors"]["IP"]["real_time_vintage"] is True

    vdh.run_prepare(
        config_path=vdh.DEFAULT_CONFIG,
        config=config,
        source_root=source.resolve(),
        paths=paths,
        resume=True,
    )
    with pytest.raises(vdh.VanDenHauweReducedError, match="create-only"):
        vdh.run_prepare(
            config_path=vdh.DEFAULT_CONFIG,
            config=config,
            source_root=source.resolve(),
            paths=paths,
            resume=False,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: row["predictors"]["IP"].update(
                {"real_time_vintage": False, "source_policy": "current_vintage"}
            ),
            "non-vintage IP",
        ),
        (
            lambda row: row["predictors"]["INF"].update(
                {"current_vintage_fallback": True}
            ),
            "current-vintage fallback for INF",
        ),
        (
            lambda row: row.update({"decision_month": False, "direction": "hold"}),
            "meeting-free month must have a null target",
        ),
    ],
)
def test_prepare_fails_closed_on_non_vintage_fallback_or_false_hold(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    config = _config()
    source, rows = _source_release(tmp_path, config)
    row = next(
        value
        for value in rows
        if value["month"] not in _h19_targets(config)
        and (value["decision_month"] is False if "meeting-free" in message else True)
    )
    mutation(row)
    _rewrite_monthly_source(source, rows)

    with pytest.raises(vdh.VanDenHauweReducedError, match=message):
        vdh.prepare_monthly_panel(config, source.resolve())


def test_source_artifact_hash_drift_is_rejected(tmp_path: Path) -> None:
    config = _config()
    source, _ = _source_release(tmp_path, config)
    with (source / "indpro_vintages.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(vdh.VanDenHauweReducedError, match="SHA256 mismatch"):
        vdh.prepare_monthly_panel(config, source.resolve())


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "config",
            "configs/main/a_different_frozen_config.json",
            "config path does not bind",
        ),
        ("config_sha256", "0" * 64, "config SHA256 does not bind"),
    ],
)
def test_source_manifest_must_bind_the_active_frozen_config(
    tmp_path: Path, field: str, replacement: str, message: str
) -> None:
    config = _config()
    source, _ = _source_release(tmp_path, config)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = replacement
    _write_json(manifest_path, manifest)

    with pytest.raises(vdh.VanDenHauweReducedError, match=message):
        vdh.prepare_monthly_panel(
            config, source.resolve(), config_path=vdh.DEFAULT_CONFIG
        )


def test_predict_and_score_use_isolated_model_boundary_without_pooling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    source, _ = _source_release(tmp_path, config)
    paths = vdh.runtime_paths(config, tmp_path / "run")
    vdh.run_prepare(
        config_path=vdh.DEFAULT_CONFIG,
        config=config,
        source_root=source.resolve(),
        paths=paths,
        resume=False,
    )

    calls: list[str] = []

    def fake_contract_predictions(_module: Any, **kwargs: Any) -> list[dict[str, Any]]:
        contract = kwargs["contract"]
        calls.append(contract)
        if contract == "matched_h19_posterior_imputation":
            targets = [row for row in kwargs["panel"] if row["matched_h19"]]
        else:
            targets = [row for row in kwargs["panel"] if row["paper_recursive_oos"]]
        return [
            {
                "month": row["month"],
                "coverage_status": "available",
                "unavailable_reason": None,
                "probabilities": {
                    direction: 1.0 if direction == row["target_direction"] else 0.0
                    for direction in vdh.DIRECTIONS
                },
                "training_count": 100,
                "posterior_diagnostics": {"synthetic": True},
            }
            for row in targets
        ]

    monkeypatch.setattr(
        vdh,
        "_load_model_module",
        lambda _config: (object(), vdh.IMPLEMENTATION),
    )
    monkeypatch.setattr(vdh, "_fit_contract_predictions", fake_contract_predictions)
    predictions = vdh.run_predict(
        config_path=vdh.DEFAULT_CONFIG,
        config=config,
        source_root=source.resolve(),
        paths=paths,
        contracts=vdh.CONTRACTS,
        seed=17,
        resume=False,
    )
    summary = vdh.run_score(
        config_path=vdh.DEFAULT_CONFIG,
        config=config,
        source_root=source.resolve(),
        paths=paths,
        contracts=vdh.CONTRACTS,
        resume=False,
    )

    assert calls == list(vdh.CONTRACTS)
    assert len(predictions) == 19 + 62
    assert summary["pooled_result"] is None
    assert summary["contracts"]["matched_h19_posterior_imputation"]["total_n"] == 19
    recursive = summary["contracts"]["paper_recursive_oos"]
    assert recursive["population_id"] == "paper_recursive_oos_full_n62"
    assert recursive["total_n"] == 62
    assert recursive["h19_overlap"]["population_id"] == (
        "paper_recursive_oos_h19_overlap_n8"
    )
    assert recursive["h19_overlap"]["is_subset_of"] == (
        "paper_recursive_oos_full_n62"
    )
    assert recursive["h19_overlap"]["pooled_with_full_oos"] is False
    assert recursive["h19_overlap"]["total_n"] == 8
    assert recursive["h19_overlap"]["target_support"] == {
        "cut": 3,
        "hold": 2,
        "hike": 3,
    }
    for contract in vdh.CONTRACTS:
        assert summary["contracts"][contract]["metrics"]["argmax_accuracy"] == 1.0
        assert (
            summary["contracts"][contract]["metrics"][
                "mean_true_direction_probability"
            ]
            == 1.0
        )
    assert recursive["h19_overlap"]["metrics"]["argmax_accuracy"] == 1.0
    assert (
        recursive["h19_overlap"]["metrics"][
            "mean_true_direction_probability"
        ]
        == 1.0
    )


def test_status_phase_does_not_require_unavailable_source_release(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = vdh.main(["--phase", "status", "--output-root", str(tmp_path / "run")])

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifacts"]["monthly_panel"]["exists"] is False
    assert payload["artifacts"]["predictions"]["exists"] is False
