from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import pytest

from jobs.eval import render_chk3_sentiment_score_closeness_report_v1 as base
from jobs.eval import render_chk3_sentiment_score_closeness_report_v2 as subject
from open_r1.validator.loo_generation_spec import validate_manifest_integrity
from tests import test_render_chk3_sentiment_score_closeness_report_v1 as fixtures


BASE_RENDERER_SHA256 = (
    "2cd552a62eb78686e186dc8d4acd0672a132e8060ffb6bd695191c7c1d43ca82"
)
REFERENCE_MEANS = {
    subject.PRIMARY_BACKEND_ID: -0.0034142132182068963,
    subject.ROBUSTNESS_BACKEND_ID: -0.2979276938967814,
}


def _report_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    monkeypatch.setattr(base, "ROOT", tmp_path)
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    model = fixtures._report_model(tmp_path)
    for backend_id, sd in fixtures.REFERENCE_SDS.items():
        model["reference_scales"][backend_id] = {
            "mean": REFERENCE_MEANS[backend_id],
            "standard_deviation": sd,
            "rows": subject.PRIMARY_N_MEETINGS,
            "ddof": 1,
            "population": "all_128_pre_external_deterministic_reference_meetings",
        }
    return model


def _tables(artifact: dict, pattern: str) -> list[dict]:
    compiled = re.compile(pattern)
    return [
        table
        for table in artifact["manifest"]["tables"]
        if compiled.fullmatch(table["id"])
    ]


def _tamper_scoped(
    html: str,
    *,
    marker: str,
    closing_tag: str,
    old: str,
    new: str,
) -> str:
    start = html.index(marker)
    end = html.index(closing_tag, start) + len(closing_tag)
    fragment = html[start:end]
    assert old in fragment
    return f"{html[:start]}{fragment.replace(old, new, 1)}{html[end:]}"


def _tamper_table_row(
    html: str,
    *,
    table_id: str,
    identity_cells: tuple[str, ...],
    old_cell: str,
    new_cell: str,
) -> str:
    marker = f'data-table-id="{table_id}"'
    start = html.index(marker)
    end = html.index("</section>", start) + len("</section>")
    fragment = html[start:end]
    rows = re.findall(r"<tr>.*?</tr>", fragment, flags=re.DOTALL)
    matches = [
        row for row in rows if all(f"<td>{cell}</td>" in row for cell in identity_cells)
    ]
    assert len(matches) == 1
    assert old_cell in matches[0]
    replacement = matches[0].replace(old_cell, new_cell, 1)
    return f"{html[:start]}{fragment.replace(matches[0], replacement, 1)}{html[end:]}"


def test_v2_artifact_preserves_exact_scales_and_complete_fallback_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _report_model(monkeypatch, tmp_path)

    artifact = subject.build_canonical_artifact(model)
    contract = subject._validate_artifact_fallback_contract(artifact)

    assert artifact["manifest"]["version"] == 1
    assert artifact["snapshot"]["version"] == 1
    assert contract == {
        "estimate_rows": 600,
        "estimate_display_rows": 600,
        "contrast_rows": 400,
        "contrast_display_rows": 400,
        "estimate_pages": 40,
        "contrast_pages": 27,
        "reference_scale_rows": 2,
        "support_rows": 8,
        "support_raw_exact_fields": 6,
        "headline_point_cards": 10,
        "scatter_rows": 256,
        "scatter_exact_fields": 3,
        "view_robustness_rows": 25,
        "manifest_id_counts": {
            "cards": 12,
            "charts": 6,
            "tables": 76,
            "blocks": 112,
            "sources": 10,
        },
        "dataset_count": 44,
        "dataset_limit": 50,
        "max_dataset_fields": 72,
        "dataset_field_limit": 80,
        "canonical_artifact_bytes": contract["canonical_artifact_bytes"],
        "artifact_byte_limit": 3_000_000,
        "table_row_limit": 15,
        "chart_row_limit": 50,
    }
    assert contract["canonical_artifact_bytes"] < 3_000_000

    estimate_tables = _tables(artifact, r"all_estimates_page_\d{2}_table")
    contrast_tables = _tables(artifact, r"all_contrasts_page_\d{2}_table")
    view_tables = _tables(artifact, r"view_robustness_page_\d{2}_table")
    assert len(estimate_tables) == 40
    assert len(contrast_tables) == 27
    assert len(view_tables) == 2
    assert len(artifact["manifest"]["tables"]) == 76
    assert len(artifact["manifest"]["charts"]) == 6
    assert all(
        len(artifact["snapshot"]["datasets"][table["dataset"]]) <= 15
        for table in artifact["manifest"]["tables"]
    )
    assert all(
        len(artifact["snapshot"]["datasets"][chart["dataset"]]) <= 50
        for chart in artifact["manifest"]["charts"]
    )

    estimate_ids = subject._appendix_ids_from_v2(
        artifact,
        prefix="all_estimates",
        row_id="estimate_id",
        expected=600,
    )
    contrast_ids = subject._appendix_ids_from_v2(
        artifact,
        prefix="all_contrasts",
        row_id="contrast_row_id",
        expected=400,
    )
    assert estimate_ids == sorted(estimate_ids)
    assert contrast_ids == sorted(contrast_ids)
    assert len(set(estimate_ids)) == 600
    assert len(set(contrast_ids)) == 400

    for backend_id, mean in REFERENCE_MEANS.items():
        row = next(
            item
            for item in artifact["snapshot"]["datasets"]["reference_scales"]
            if item["backend_id"] == backend_id
        )
        assert row["reference_mean_display"] == f"{mean!r} (exact)"
        assert row["reference_sd_display"] == (
            f"{fixtures.REFERENCE_SDS[backend_id]!r} (exact)"
        )
        with pytest.raises(ValueError):
            float(row["reference_mean_display"])
        with pytest.raises(ValueError):
            float(row["reference_sd_display"])

    support_rows = artifact["snapshot"]["datasets"]["support_diagnostics"]
    assert len(support_rows) == 8
    for row in support_rows:
        for source_field, display_field in subject.SUPPORT_EXACT_FIELDS.items():
            assert row[display_field] == f"{float(row[source_field])!r} (exact)"
            with pytest.raises(ValueError):
                float(row[display_field])
    support_table = next(
        table
        for table in artifact["manifest"]["tables"]
        if table["id"] == "support_diagnostics_table"
    )
    support_columns = {column["field"]: column for column in support_table["columns"]}
    for source_field, display_field in subject.SUPPORT_EXACT_FIELDS.items():
        assert source_field not in support_columns
        assert support_columns[display_field]["type"] == "text"
        assert "format" not in support_columns[display_field]

    headline = artifact["snapshot"]["datasets"]["headline"][0]
    assert headline["spearman_estimate"] == "+0.003908 (6 d.p.)"
    assert len(subject._headline_point_contract(artifact)) == 10
    assert (
        sum(
            len(rows)
            for rows in subject._appendix_displays_by_table(
                artifact,
                prefix="all_estimates",
                row_id="estimate_id",
                numeric_field="estimate",
                display_field="estimate_display",
                expected=600,
            ).values()
        )
        == 600
    )
    assert (
        sum(
            len(rows)
            for rows in subject._appendix_displays_by_table(
                artifact,
                prefix="all_contrasts",
                row_id="contrast_row_id",
                numeric_field="gain",
                display_field="gain_display",
                expected=400,
            ).values()
        )
        == 400
    )

    for prefix in ("distil_raw_scatter", "finbert_raw_scatter"):
        page_names = [
            name
            for name in artifact["snapshot"]["datasets"]
            if re.fullmatch(rf"{prefix}_page_\d{{2}}", name)
        ]
        assert sorted(
            len(artifact["snapshot"]["datasets"][name]) for name in page_names
        ) == [28, 50, 50]
        meetings = [
            row["meeting_id"]
            for name in sorted(page_names)
            for row in artifact["snapshot"]["datasets"][name]
        ]
        assert len(meetings) == len(set(meetings)) == 128
        for name in page_names:
            for row in artifact["snapshot"]["datasets"][name]:
                for source_field, display_field in subject.SCATTER_EXACT_FIELDS.items():
                    assert row[display_field] == (
                        f"{float(row[source_field])!r} (exact)"
                    )
    assert (
        sum(len(rows) for rows in subject._view_ids_by_table(artifact).values()) == 25
    )
    assert subject._binding(Path(base.__file__))["sha256"] == BASE_RENDERER_SHA256

    bad_headline = copy.deepcopy(artifact)
    bad_headline["snapshot"]["datasets"]["headline"][0]["spearman_estimate"] = (
        "+9.999999 (6 d.p.)"
    )
    with pytest.raises(
        subject.SentimentClosenessReportError, match="authoritative point"
    ):
        subject._validate_artifact_fallback_contract(bad_headline)

    bad_view = copy.deepcopy(artifact)
    bad_view["snapshot"]["datasets"]["view_robustness_page_01"][0]["gain_display"] = (
        "+9.999999 (6 d.p.)"
    )
    with pytest.raises(
        subject.SentimentClosenessReportError, match="authoritative gain"
    ):
        subject._validate_artifact_fallback_contract(bad_view)

    duplicate_card = copy.deepcopy(artifact)
    duplicate_card["manifest"]["cards"].append(
        copy.deepcopy(duplicate_card["manifest"]["cards"][0])
    )
    with pytest.raises(subject.SentimentClosenessReportError, match="duplicated"):
        subject._validate_artifact_fallback_contract(duplicate_card)

    extra_scatter = copy.deepcopy(artifact)
    extra_chart = copy.deepcopy(extra_scatter["manifest"]["charts"][0])
    extra_chart.update(
        {
            "id": "distil_raw_scatter_page_99_chart",
            "dataset": "distil_raw_scatter_page_99",
        }
    )
    extra_scatter["manifest"]["charts"].append(extra_chart)
    extra_scatter["snapshot"]["datasets"]["distil_raw_scatter_page_99"] = []
    with pytest.raises(
        subject.SentimentClosenessReportError, match="chart ID inventory"
    ):
        subject._validate_artifact_fallback_contract(extra_scatter)


def test_v2_packaging_is_fail_closed_and_seals_both_renderers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    node = repository_root / ".cache/report_node/bin/node"
    if not node.is_file():
        pytest.skip("frozen local Node runtime unavailable")
    if not base._delivery.DEFAULT_PLUGIN_ROOT.is_dir():
        pytest.skip("packaged Data Analytics report builder unavailable")
    model = _report_model(monkeypatch, tmp_path)
    estimator_binding = model["source_bindings"]["estimator_manifest"]
    load_calls = 0
    package_calls = 0

    def fake_load(*_args: object, **_kwargs: object) -> tuple[dict, dict]:
        nonlocal load_calls
        load_calls += 1
        return copy.deepcopy(model), copy.deepcopy(estimator_binding)

    real_package = subject._package_report

    def counted_package(**kwargs: object) -> tuple[dict, dict]:
        nonlocal package_calls
        package_calls += 1
        return real_package(**kwargs)

    monkeypatch.setattr(subject, "load_report_model", fake_load)
    monkeypatch.setattr(subject, "_package_report", counted_package)
    output = tmp_path / "sentiment_score_closeness_report_v2"

    manifest = subject.render_and_seal_report(
        estimator_manifest=Path(estimator_binding["path"]),
        estimator_manifest_sha256=estimator_binding["sha256"],
        output_dir=output,
        node_bin=node,
        plugin_root=base._delivery.DEFAULT_PLUGIN_ROOT,
    )

    assert load_calls == 2
    assert package_calls == 1
    assert (
        validate_manifest_integrity(manifest) == manifest["integrity"]["payload_sha256"]
    )
    assert manifest["schema_version"] == subject.REPORT_MANIFEST_SCHEMA
    assert manifest["report_id"] == subject.REPORT_ID
    assert manifest["artifacts"]["canonical_artifact"]["contract"] == (
        subject.REPORT_ARTIFACT_CONTRACT
    )
    assert (
        manifest["renderer"]["sha256"]
        == subject._binding(Path(subject.__file__))["sha256"]
    )
    assert manifest["base_renderer"]["sha256"] == BASE_RENDERER_SHA256
    assert manifest["report_contract"]["appendix_page_size"] == 15
    assert manifest["report_contract"]["fallback_fidelity"]["dataset_count"] == 44
    html_acceptance = manifest["report_contract"]["html_acceptance"]
    assert html_acceptance["showing_first_occurrences"] == 0
    assert html_acceptance["visible_host_counts"] == {
        "tables": 76,
        "charts": 6,
        "cards": 12,
    }
    assert html_acceptance["visible_estimate_rows"] == 600
    assert html_acceptance["unique_visible_estimate_rows"] == 600
    assert html_acceptance["visible_estimate_pages"] == 40
    assert html_acceptance["visible_estimate_exact_displays"] == 600
    assert len(html_acceptance["sorted_estimate_ids_sha256"]) == 64
    assert html_acceptance["visible_contrast_rows"] == 400
    assert html_acceptance["unique_visible_contrast_rows"] == 400
    assert html_acceptance["visible_contrast_pages"] == 27
    assert html_acceptance["visible_contrast_exact_displays"] == 400
    assert len(html_acceptance["sorted_contrast_ids_sha256"]) == 64
    assert html_acceptance["visible_reference_scale_rows"] == 2
    assert html_acceptance["reference_scale_exact_text"] is True
    assert html_acceptance["visible_headline_point_cards"] == 10
    assert html_acceptance["headline_point_exact_text"] is True
    assert html_acceptance["visible_distil_scatter_rows"] == 128
    assert html_acceptance["visible_finbert_scatter_rows"] == 128
    assert html_acceptance["visible_scatter_exact_cells"] == 768
    assert html_acceptance["visible_view_robustness_rows"] == 25
    assert html_acceptance["visible_support_rows"] == 8
    assert html_acceptance["visible_support_exact_cells"] == 48
    assert html_acceptance["support_raw_exact_text"] is True
    assert (
        html_acceptance["html_sha256"]
        == manifest["artifacts"]["portable_report"]["sha256"]
    )
    assert {path.name for path in output.iterdir()} == {
        "artifact.json",
        "report.html",
        "manifest.json",
    }
    assert output.stat().st_mode & 0o222 == 0
    assert all(path.stat().st_mode & 0o222 == 0 for path in output.iterdir())

    artifact = copy.deepcopy(subject.build_canonical_artifact(model))
    report_path = output / "report.html"
    acceptance = subject.validate_packaged_html(report_path, artifact)
    assert acceptance == html_acceptance
    html = report_path.read_text(encoding="utf-8")
    assert "Showing first" not in html
    assert f"{REFERENCE_MEANS[subject.PRIMARY_BACKEND_ID]!r} (exact)" in html
    assert f"{fixtures.REFERENCE_SDS[subject.PRIMARY_BACKEND_ID]!r} (exact)" in html
    assert "+0.003908 (6 d.p.)" in html

    bad_truncation = tmp_path / "bad_truncation.html"
    bad_truncation.write_text(
        html.replace("</body>", "<p>Showing first</p></body>", 1),
        encoding="utf-8",
    )
    with pytest.raises(subject.SentimentClosenessReportError, match="truncated"):
        subject.validate_packaged_html(bad_truncation, artifact)

    removable_id = next(
        row["estimate_id"]
        for row in model["estimates"]
        if row["backend_id"] == subject.ROBUSTNESS_BACKEND_ID
        and row["view_id"] == "full"
        and row["score_policy"] == "neutral_imputed_fixed_k5"
        and row["scale_id"] == "raw_score"
        and row["arm"] == "chk0"
        and row["metric_id"] == "pearson_r"
    )
    visible_cell = f"<td>{removable_id}</td>"
    assert html.count(visible_cell) == 1
    bad_identity = tmp_path / "bad_identity.html"
    bad_identity.write_text(
        html.replace(visible_cell, "<td>missing-estimate-id</td>", 1),
        encoding="utf-8",
    )
    with pytest.raises(subject.SentimentClosenessReportError, match="incomplete"):
        subject.validate_packaged_html(bad_identity, artifact)

    exact_mean = f"{REFERENCE_MEANS[subject.PRIMARY_BACKEND_ID]!r} (exact)"
    bad_precision = tmp_path / "bad_precision.html"
    bad_precision.write_text(html.replace(exact_mean, "-0", 1), encoding="utf-8")
    with pytest.raises(subject.SentimentClosenessReportError, match="exact reference"):
        subject.validate_packaged_html(bad_precision, artifact)

    distil_support = next(
        row
        for row in artifact["snapshot"]["datasets"]["support_diagnostics"]
        if row["backend_id"] == subject.PRIMARY_BACKEND_ID
        and row["artifact_arm"] == subject.REFERENCE_ARM
    )
    support_mean_cell = f"<td>{distil_support['mean_raw_score_display']}</td>"
    bad_support = tmp_path / "bad_support_precision.html"
    bad_support.write_text(
        _tamper_table_row(
            html,
            table_id="support_diagnostics_table",
            identity_cells=(distil_support["backend"], subject.REFERENCE_ARM),
            old_cell=support_mean_cell,
            new_cell="<td>-0</td>",
        ),
        encoding="utf-8",
    )
    with pytest.raises(subject.SentimentClosenessReportError, match="compacted Distil"):
        subject.validate_packaged_html(bad_support, artifact)

    estimate_table = _tables(artifact, r"all_estimates_page_\d{2}_table")[0]
    estimate_fields = [column["field"] for column in estimate_table["columns"]]
    estimate_display_field = next(
        field for field in estimate_fields if field.endswith("__estimate_display")
    )
    estimate_display = artifact["snapshot"]["datasets"][estimate_table["dataset"]][0][
        estimate_display_field
    ]
    bad_estimate = tmp_path / "bad_estimate_precision.html"
    bad_estimate.write_text(
        _tamper_scoped(
            html,
            marker=f'data-table-id="{estimate_table["id"]}"',
            closing_tag="</section>",
            old=f"<td>{estimate_display}</td>",
            new="<td>0</td>",
        ),
        encoding="utf-8",
    )
    with pytest.raises(subject.SentimentClosenessReportError, match="estimate display"):
        subject.validate_packaged_html(bad_estimate, artifact)

    headline_value = artifact["snapshot"]["datasets"]["headline"][0][
        "spearman_estimate"
    ]
    bad_headline = tmp_path / "bad_headline_precision.html"
    bad_headline.write_text(
        _tamper_scoped(
            html,
            marker='data-card-id="headline_spearman_direct_card"',
            closing_tag="</article>",
            old=(f'<span class="portable-source-value-text">{headline_value}</span>'),
            new='<span class="portable-source-value-text">0</span>',
        ),
        encoding="utf-8",
    )
    with pytest.raises(subject.SentimentClosenessReportError, match="headline point"):
        subject.validate_packaged_html(bad_headline, artifact)

    scatter_chart = next(
        chart
        for chart in artifact["manifest"]["charts"]
        if chart["id"] == "distil_raw_scatter_page_01_chart"
    )
    scatter_row = artifact["snapshot"]["datasets"][scatter_chart["dataset"]][0]
    scatter_value = scatter_row["reference_raw_score_display"]
    bad_scatter = tmp_path / "bad_scatter_precision.html"
    bad_scatter.write_text(
        _tamper_scoped(
            html,
            marker=f'data-chart-id="{scatter_chart["id"]}"',
            closing_tag="</figure>",
            old=f"<td>{scatter_value}</td>",
            new="<td>0</td>",
        ),
        encoding="utf-8",
    )
    with pytest.raises(subject.SentimentClosenessReportError, match="scatter value"):
        subject.validate_packaged_html(bad_scatter, artifact)

    view_table = _tables(artifact, r"view_robustness_page_\d{2}_table")[0]
    view_id = artifact["snapshot"]["datasets"][view_table["dataset"]][0][
        "contrast_row_id"
    ]
    bad_view = tmp_path / "bad_view_identity.html"
    bad_view.write_text(
        _tamper_scoped(
            html,
            marker=f'data-table-id="{view_table["id"]}"',
            closing_tag="</section>",
            old=f"<td>{view_id}</td>",
            new="<td>missing-view-id</td>",
        ),
        encoding="utf-8",
    )
    with pytest.raises(subject.SentimentClosenessReportError, match="view identity"):
        subject.validate_packaged_html(bad_view, artifact)

    fixtures._make_writable(output)


def test_v2_create_only_rejects_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "sentiment_score_closeness_report_v2"
    output.mkdir()
    with pytest.raises(subject.SentimentClosenessReportError, match="already exists"):
        subject.render_and_seal_report(
            estimator_manifest=tmp_path / "unused.json",
            estimator_manifest_sha256="0" * 64,
            output_dir=output,
        )
    assert os.access(output, os.W_OK)


def test_v2_html_acceptance_failure_removes_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _report_model(monkeypatch, tmp_path)
    estimator_binding = model["source_bindings"]["estimator_manifest"]
    load_calls = 0

    def fake_load(*_args: object, **_kwargs: object) -> tuple[dict, dict]:
        nonlocal load_calls
        load_calls += 1
        return copy.deepcopy(model), copy.deepcopy(estimator_binding)

    def truncated_package(**kwargs: object) -> tuple[dict, dict]:
        report_path = kwargs["report_path"]
        assert isinstance(report_path, Path)
        report_path.write_text(
            "<!doctype html><html><body>Showing first 15</body></html>\n",
            encoding="utf-8",
        )
        return {}, {}

    monkeypatch.setattr(subject, "load_report_model", fake_load)
    monkeypatch.setattr(subject, "_package_report", truncated_package)
    output = tmp_path / "sentiment_score_closeness_report_v2"

    with pytest.raises(subject.SentimentClosenessReportError, match="truncated"):
        subject.render_and_seal_report(
            estimator_manifest=Path(estimator_binding["path"]),
            estimator_manifest_sha256=estimator_binding["sha256"],
            output_dir=output,
        )

    assert load_calls == 1
    assert not output.exists()
    assert not list(tmp_path.glob(".sentiment_score_closeness_report_v2.staging-*"))
