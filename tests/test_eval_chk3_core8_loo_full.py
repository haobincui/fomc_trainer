from __future__ import annotations

import json
from pathlib import Path

from jobs.eval import eval_chk3_core8_loo_full as full


ROOT = Path(__file__).resolve().parents[1]


def test_full_config_has_exact_n128_matrix() -> None:
    config = json.loads((ROOT / full.DEFAULT_CONFIG).read_text(encoding="utf-8"))
    assert config["schema_version"] == full.CONFIG_SCHEMA
    assert config["evaluation_id"] == full.EVALUATION_ID
    assert config["meeting_selection"]["count"] == 128
    assert config["expected_generation_rows"] == 128 * 17 == 2176
    assert config["scope"]["primary_result"].startswith("raw paired")


def test_full_panel_selects_every_meeting_chronologically() -> None:
    panel = ROOT / (
        "dataset/processed/retrain_v2/"
        "chk3_minutes_external_holdout_1993_2008_all_regular_v1/panels/core8.jsonl"
    )
    rows = [json.loads(line) for line in panel.read_text(encoding="utf-8").splitlines()]
    selected = full.select_all_meetings(rows, [[1993, 2008]])
    assert len(selected) == len(set(selected)) == 128
    dates = {
        str(row["meeting_id"]): str(row["meeting_start_date"])
        for row in rows
    }
    assert [dates[meeting_id] for meeting_id in selected] == sorted(dates.values())


def test_full_specialization_sets_versioned_contract() -> None:
    full.configure_implementation()
    impl = full.implementation
    assert impl.EXPECTED_MEETINGS == 128
    assert impl.EXPECTED_ROWS == 2176
    assert impl.EVALUATION_ID == full.EVALUATION_ID
    assert impl.GPU_LOCK == full.GPU_LOCK
