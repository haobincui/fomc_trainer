from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "jobs/eval/evaluate_yoon_fan_ordinal_forest_reduced.py"
SPEC = importlib.util.spec_from_file_location("yoon_fan_reduced", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def load_config() -> dict:
    return json.loads(
        (ROOT / "configs/main/yoon_fan_ordinal_forest_reduced_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_configuration_and_source_inputs_close() -> None:
    config = load_config()
    module.validate_config(config)
    source = ROOT / config["data_contract"]["source_root"]
    train, test, metadata = module.build_reduced_inputs(source, config)
    assert len(train) == 211
    assert len(test) == 19
    assert len(metadata) == 230
    assert {row["direction"] for row in train} == {"cut", "hold", "hike"}
    assert all("direction" not in row for row in test)
    assert sum(not metadata[row["row_id"]]["original_paper_sample_eligible"] for row in test) == 2
    h19_end = max(metadata[row["row_id"]]["meeting_start_date"] for row in test)
    assert (
        sum(metadata[row["row_id"]]["meeting_start_date"] > h19_end for row in train)
        == 106
    )


def test_r_test_csv_excludes_target(tmp_path: Path) -> None:
    train = [
        {
            "row_id": "train-1",
            "direction": "hold",
            "tbill6m_minus_effr_pp": 0.1,
            "unemployment_rate_pct": 5.0,
            "real_gdp_growth_annualized_pct": 2.0,
        }
    ]
    test = [
        {
            "row_id": "test-1",
            "tbill6m_minus_effr_pp": 0.2,
            "unemployment_rate_pct": 5.1,
            "real_gdp_growth_annualized_pct": 2.1,
        }
    ]
    train_path, test_path = module.write_r_inputs(tmp_path / "inputs", train, test)
    with train_path.open(newline="", encoding="utf-8") as handle:
        assert "direction" in next(csv.reader(handle))
    with test_path.open(newline="", encoding="utf-8") as handle:
        assert "direction" not in next(csv.reader(handle))


def test_load_r_predictions_enforces_point_only_contract(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    path.write_text(
        "row_id,predicted_direction,probability_status\n"
        "a,cut,unavailable_equal_performance_author_configuration\n"
        "b,hold,unavailable_equal_performance_author_configuration\n",
        encoding="utf-8",
    )
    assert module.load_r_predictions(path, ["a", "b"]) == {"a": "cut", "b": "hold"}

    path.write_text(
        "row_id,predicted_direction,probability_status\n"
        "a,cut,available\n",
        encoding="utf-8",
    )
    with pytest.raises(module.YoonFanReplicationError, match="probability"):
        module.load_r_predictions(path, ["a"])


def test_point_metrics_are_recomputed_at_meeting_grain() -> None:
    truth = {
        "c1": "cut",
        "c2": "cut",
        "h1": "hold",
        "h2": "hold",
        "u1": "hike",
    }
    predicted = {
        "c1": "cut",
        "c2": "hold",
        "h1": "hold",
        "h2": "hike",
        "u1": "hike",
    }
    result = module.score_point_predictions(truth, predicted)
    assert result["overall_accuracy"] == pytest.approx(3 / 5)
    assert result["balanced_accuracy"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert result["per_class"]["cut"]["correct"] == 1
    assert result["per_class"]["hold"]["support"] == 2
    assert result["per_class"]["hike"]["accuracy"] == 1.0
    assert result["probability_metrics"] is None


def test_config_rejects_probability_mode() -> None:
    config = load_config()
    config["model"]["parameters"]["perffunction"] = "probability"
    with pytest.raises(module.YoonFanReplicationError, match="model contract"):
        module.validate_config(config)


def test_package_tree_hash_binds_names_and_contents(tmp_path: Path) -> None:
    package = tmp_path / "ordinalForest"
    package.mkdir()
    (package / "DESCRIPTION").write_text("Version: 2.4-3\n", encoding="utf-8")
    first = module.sha256_tree(package)
    (package / "DESCRIPTION").write_text("Version: 2.4-2\n", encoding="utf-8")
    assert module.sha256_tree(package) != first
