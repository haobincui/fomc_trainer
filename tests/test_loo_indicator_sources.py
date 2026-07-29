from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "configs/main/loo_indicator_sources.json"
ROSTER_PATH = REPO_ROOT / "configs/main/leave_one_out_roster.json"

REQUIRED_SOURCE_FIELDS = {
    "source_id",
    "artifact_id",
    "series_id",
    "provider",
    "access_interface",
    "label",
    "title",
    "frequency",
    "units",
    "seasonal_adjustment",
    "lookback_years",
    "lookback",
    "transformation",
    "license",
    "redistribution_allowed",
    "enabled",
}

ICE_SERIES = {
    "BAMLC1A0C13YEY",
    "BAMLC7A0C1015YEY",
    "BAMLC8A0C15PYEY",
    "BAMLC2A0C35YEY",
    "BAMLC3A0C57YEY",
    "BAMLC4A0C710YEY",
    "BAMLC0A2CAAEY",
    "BAMLC0A1CAAAEY",
    "BAMLEM4RBLLCRPIUSEY",
    "BAMLH0A1HYBBEY",
    "BAMLC0A4CBBBEY",
    "BAMLH0A3HYCEY",
}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _crosswalk_by_path(registry: dict) -> dict[str, dict]:
    return {row["path"]: row for row in registry["legacy_crosswalk"]["files"]}


def test_registry_freezes_keyless_d1_policy_and_exact_roster() -> None:
    registry = _load(REGISTRY_PATH)
    roster = _load(ROSTER_PATH)

    assert registry["schema_version"] == "loo-indicator-source-registry-v1"
    assert registry["roster_id"] == roster["roster_id"]
    assert list(registry["indicators"]) == roster["indicators"]

    policy = registry["policy"]
    assert policy["network_mode"] == "keyless"
    assert policy["vintage_lag_calendar_days"] == 1
    assert policy["same_meeting_day_data"] == "excluded"
    assert policy["unknown_availability"] == "fail_closed"
    assert policy["runtime_fallback"] == "forbidden"
    assert policy["local_legacy_values"] == "mapping_only"
    assert policy["synthetic_text_values"] == "mapping_only"
    assert policy["access_interface"] == "alfred-graph-csv-v1"
    assert "api_key" not in policy["request_url_template"].lower()
    assert policy["raw_response_required"] is True
    assert policy["raw_response_sha256_required"] is True
    assert policy["restricted_artifacts_git_policy"] == "gitignored"
    assert policy["ledger_git_policy"] == "gitignored"


def test_every_source_has_explicit_fetch_and_provenance_metadata() -> None:
    registry = _load(REGISTRY_PATH)
    sources = registry["sources"]

    assert len(sources) >= 100
    assert len({source["series_id"] for source in sources.values()}) == len(sources)
    for source_key, source in sources.items():
        assert REQUIRED_SOURCE_FIELDS <= source.keys(), source_key
        assert source_key == source["artifact_id"]
        assert source_key == f"alfred__{source['series_id']}"
        assert source["source_id"] == f"alfred:{source['series_id']}"
        assert source["provider"] == "ALFRED"
        assert source["access_interface"] == "alfred-graph-csv-v1"
        assert source["title"] == source["label"]
        assert source["seasonal_adjustment"]
        assert source["lookback_years"] == 2
        assert source["lookback"] == {"unit": "months", "value": 24}
        assert source["transformation"] == {"method": "identity"}
        assert source["license"]
        assert isinstance(source["redistribution_allowed"], bool)
        assert isinstance(source["enabled"], bool)
        if not source["enabled"]:
            assert source["disabled_reason"] == "keyless_alfred_vintage_http_404"


def test_each_indicator_has_a_known_enabled_source() -> None:
    registry = _load(REGISTRY_PATH)
    sources = registry["sources"]

    for indicator, entry in registry["indicators"].items():
        source_keys = entry["source_keys"]
        assert source_keys, indicator
        assert len(source_keys) == len(set(source_keys)), indicator
        assert set(source_keys) <= sources.keys(), indicator
        assert any(sources[key]["enabled"] for key in source_keys), indicator


def test_disabled_sources_are_only_verified_keyless_vintage_failures() -> None:
    registry = _load(REGISTRY_PATH)
    sources = registry["sources"]
    disabled_ids = {
        source["series_id"] for source in sources.values() if not source["enabled"]
    }

    assert disabled_ids == ICE_SERIES | {"ADPWNUSNERSA", "SP500", "DJIA"}
    assert registry["sources"]["alfred__USPRIV"]["enabled"] is True
    assert registry["sources"]["alfred__NFCICREDIT"]["enabled"] is True
    assert registry["sources"]["alfred__SPASTT01USM661N"]["enabled"] is True
    assert registry["sources"]["alfred__NFCIRISK"]["enabled"] is True


def test_crosswalk_covers_exactly_the_102_local_legacy_files() -> None:
    registry = _load(REGISTRY_PATH)
    legacy = registry["legacy_crosswalk"]
    legacy_root = REPO_ROOT / legacy["root"]
    actual_paths = {
        path.relative_to(legacy_root).as_posix()
        for path in legacy_root.rglob("*")
        if path.is_file()
    }
    rows = legacy["files"]
    configured_paths = [row["path"] for row in rows]

    assert legacy["expected_file_count"] == 102
    assert len(actual_paths) == 102
    assert len(rows) == 102
    assert len(configured_paths) == len(set(configured_paths))
    assert set(configured_paths) == actual_paths
    assert legacy["value_reuse_allowed"] is False
    assert legacy["mapping_reuse_allowed"] is True

    sources = registry["sources"]
    for row in rows:
        assert row["indicator"] in registry["indicators"]
        assert row["disposition"]
        assert row["reason"]
        assert row["source_keys"]
        assert set(row["source_keys"]) <= set(
            registry["indicators"][row["indicator"]]["source_keys"]
        )
        assert set(row["source_keys"]) <= sources.keys()
        assert any(sources[key]["enabled"] for key in row["source_keys"])


def test_high_risk_legacy_files_have_prespecified_replacements() -> None:
    registry = _load(REGISTRY_PATH)
    rows = _crosswalk_by_path(registry)

    adp_path = (
        "Labour Market/"
        "Total Nonfarm Private Payroll Employment(Persons, Seasonally Adjusted).csv"
    )
    assert rows[adp_path]["legacy_source_id"] == "ADPWNUSNERSA"
    assert rows[adp_path]["source_keys"] == [
        "alfred__ADPWNUSNERSA",
        "alfred__USPRIV",
    ]

    corporate_rows = [
        row
        for row in rows.values()
        if row["indicator"] == "Corporate-Bond-Yields"
        and row["legacy_source_id"] in ICE_SERIES
    ]
    assert len(corporate_rows) == len(ICE_SERIES)
    assert all("alfred__NFCICREDIT" in row["source_keys"] for row in corporate_rows)

    international_proxy = {
        "alfred__SPASTT01JPM661N",
        "alfred__SPASTT01EZM661N",
        "alfred__SPASTT01GBM661N",
    }
    for path in (
        "International Equity Markets/MSCI WORLD.csv",
        "International Equity Markets/MSCI Emerging Markets Index(future).csv",
    ):
        assert set(rows[path]["source_keys"]) == international_proxy

    assert rows[
        "Consumer Price Index (CPI)/"
        "Consumer Price Index for All Urban Consumers (CPI-U, seasonal adjusted).csv"
    ]["source_keys"] == ["alfred__CPIAUCSL"]
    assert rows[
        "Unemployment Rate/Unemployment Rate (Seasonal adjusted).csv"
    ]["source_keys"] == ["alfred__UNRATE"]


def test_restricted_sources_are_marked_without_disabling_local_analysis() -> None:
    registry = _load(REGISTRY_PATH)
    sources = registry["sources"]

    for source_id in ("UMCSENT", "AAA", "DBAA", "NASDAQCOM", "VIXCLS"):
        source = sources[f"alfred__{source_id}"]
        assert source["enabled"] is True
        assert source["redistribution_allowed"] is False
        assert source["license"] == "third_party_restricted"
