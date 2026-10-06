from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from jobs.eval import run_tadle_form_ff_futures_v1 as diagnostic
from jobs.eval import validate_tadle_form_ff_futures_v1 as validator


def test_clean_statement_html_retains_modern_release_body() -> None:
    raw = b"""
    <html><body><div id="article"><div class="heading">Navigation</div>
    <div class="col-xs-12 col-sm-8 col-md-8">
    <p>For immediate release</p>
    <p>The Federal Open Market Committee decided to maintain its policy stance.</p>
    <p>""" + (b"Economic conditions and Committee discussion. " * 12) + b"""</p>
    <div id="lastUpdate">Last Update: today</div></div></div></body></html>
    """
    text, metadata = diagnostic.clean_statement_html(raw)
    assert "Federal Open Market Committee" in text
    assert "For immediate release" not in text
    assert "Last Update" not in text
    assert metadata["selector"] == "#article substantive column"
    assert metadata["text_sha256"] == diagnostic.sha256_text(text)


def test_clean_statement_html_retains_legacy_table_body() -> None:
    raw = b"""
    <html><body><table><tr><td>Site navigation</td><td>
    For immediate release The Federal Open Market Committee decided to maintain
    the intended federal funds rate. """ + (b"Committee economic assessment. " * 12) + b"""
    </td></tr></table></body></html>
    """
    text, metadata = diagnostic.clean_statement_html(raw)
    assert "Federal Open Market Committee" in text
    assert metadata["selector"] == "largest Committee-bearing td"
    assert len(text) >= 200


def test_futures_calendar_month_mapping() -> None:
    date = pd.Timestamp("2011-12-30")
    assert diagnostic.add_calendar_month(date, 1) == pd.Period("2012-01", freq="M")
    assert diagnostic.add_calendar_month(date, 12) == pd.Period("2012-12", freq="M")


def test_tadle_design_and_hc1_fit_recover_known_news_coefficient() -> None:
    news = np.asarray([0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5])
    vix = np.asarray([0.2, -0.1, 0.3, 0.0, -0.2, 0.1, -0.3, 0.4])
    years = ["2010"] * 4 + ["2011"] * 4
    post = np.asarray([0.0] * 4 + [1.0] * 4)
    matrix, names = diagnostic.design_matrix(news, vix, years, post, interaction=False)
    y = 0.7 + 2.5 * news - 0.4 * vix + 0.2 * (np.asarray(years) == "2011")
    result = diagnostic.fit_ols_hc1(y, matrix, names)
    assert math.isclose(result["coefficients"][names.index("news_shock")], 2.5, abs_tol=1e-12)
    assert result["nobs"] == 8
    assert result["covariance"].shape == (4, 4)


def test_bootstrap_plan_is_deterministic_and_pairs_replicates() -> None:
    sample = {
        "estimation_current": [
            {"generation_meeting_id": "2005-01-01"},
            {"generation_meeting_id": "2005-02-01"},
        ],
        "needed_ids": ("2004-12-01", "2005-01-01", "2005-02-01"),
    }
    first = diagnostic.build_bootstrap_plan(sample, draws=4, seed=123)
    second = diagnostic.build_bootstrap_plan(sample, draws=4, seed=123)
    assert first["plan_sha256"] == second["plan_sha256"]
    assert np.array_equal(first["sampled_year_indexes"], second["sampled_year_indexes"])
    assert np.array_equal(first["replicate_choices"], second["replicate_choices"])
    assert first["replicate_choices"].shape == (4, 3, 5)


def test_validator_rejects_nonfinite_json_and_numeric_comparisons() -> None:
    with pytest.raises(validator.ValidationError):
        validator.strict_json_loads('{"value": NaN}')
    with pytest.raises(validator.ValidationError):
        validator.assert_finite_close("test", float("nan"), 0.0)
    with pytest.raises(validator.ValidationError):
        validator.sign_p(np.asarray([0.0, float("inf")]))

