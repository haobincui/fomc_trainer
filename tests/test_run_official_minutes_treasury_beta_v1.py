from __future__ import annotations

import math

from transformers import AutoTokenizer

from jobs.eval import run_official_minutes_treasury_beta_v1 as benchmark


def test_clean_official_html_retains_article_and_removes_chrome() -> None:
    raw = b"""
    <html><head><style>.x {}</style><script>bad()</script></head><body>
      <nav>Navigation</nav><div id="content"><div id="article">
      <h3>Minutes of the Federal Open Market Committee</h3>
      <p>Official policy discussion. Federal Open Market Committee.</p>
      <p>""" + (b"substantive text " * 100) + b"""</p>
      </div><div id="lastUpdate">Last Update</div></div><footer>Footer</footer>
    </body></html>
    """
    text, metadata = benchmark.clean_official_html(raw)
    assert metadata["selector"] == "#article"
    assert "Official policy discussion" in text
    assert "Navigation" not in text
    assert "Footer" not in text
    assert "bad()" not in text
    assert metadata["text_sha256"] == benchmark.sha256_text(text)


def test_windowing_covers_every_token_and_corrects_overlap() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        benchmark.BACKENDS[benchmark.DISTIL_ID], local_files_only=True, use_fast=True
    )
    text = "policy inflation employment output " * 700
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    specs = benchmark.window_specs(tokenizer, text)
    assert len(specs) > 1
    assert specs[0]["token_start"] == 0
    assert specs[-1]["token_end"] == token_count
    assert all(len(row["input_ids"]) <= 512 for row in specs)
    assert math.isclose(
        sum(row["aggregation_weight"] for row in specs), token_count, abs_tol=1e-8
    )


def test_model_minus_official_and_distance_gain_directions() -> None:
    betas = {"official": 2.0, "chk0": -1.0, "chk1": 1.0, "chk3": 1.8}
    contrasts, distances, gains = benchmark.contrast_values(betas)
    assert contrasts == {
        "chk0_minus_official": -3.0,
        "chk1_minus_official": -1.0,
        "chk3_minus_official": -0.19999999999999996,
    }
    assert distances["chk3"] < distances["chk1"] < distances["chk0"]
    assert gains["chk2_vs_chk1"] > 0
    assert gains["chk2_vs_chk0"] > 0
