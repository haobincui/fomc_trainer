"""Render the versioned-v2 CHK2 sentiment-score closeness report.

V2 is a presentation-only adapter over the frozen v1 report model.  It fixes
two portable-HTML fidelity defects without changing any estimator row:

* every reader-visible table has at most 15 rows and every scatter panel has
  at most 50 rows, matching the portable semantic-fallback limits; and
* frozen reference means and sample SDs are rendered as explicitly exact,
  non-numeric text so the shared renderer cannot compact-format them to zero.

The authoritative estimator is still loaded and replayed twice.  Packaging is
still create-only, staged, sealed, and atomically published.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

try:
    from jobs.eval import render_chk3_sentiment_score_closeness_report_v1 as _base
except ModuleNotFoundError as exc:  # Direct ``python jobs/eval/...py`` execution.
    if exc.name != "jobs":
        raise
    import render_chk3_sentiment_score_closeness_report_v1 as _base
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / (
    "output/evaluation/main/chk3_beta_core8_merged_n2048_vllm_k5_20260816_v1"
)
DEFAULT_ESTIMATOR_MANIFEST = (
    RUN_ROOT / "sentiment_score_closeness_bootstrap_v1/manifest.json"
)
DEFAULT_OUTPUT_ROOT = RUN_ROOT / "sentiment_score_closeness_report_v2"

ESTIMATOR_MANIFEST_SCHEMA = _base.ESTIMATOR_MANIFEST_SCHEMA
REPORT_MANIFEST_SCHEMA = "chk3-sentiment-score-closeness-report-manifest-v2"
REPORT_ARTIFACT_CONTRACT = "chk3-sentiment-score-closeness-report-artifact-v2"
REPORT_ID = "chk3-sentiment-score-closeness-report-v2"

EXPECTED_BOOTSTRAP_DRAWS = _base.EXPECTED_BOOTSTRAP_DRAWS
PRIMARY_N_MEETINGS = _base.PRIMARY_N_MEETINGS
PRIMARY_BACKEND_ID = _base.PRIMARY_BACKEND_ID
ROBUSTNESS_BACKEND_ID = _base.ROBUSTNESS_BACKEND_ID
PRIMARY_VIEW_ID = _base.PRIMARY_VIEW_ID
PRIMARY_SCORE_POLICY = _base.PRIMARY_SCORE_POLICY
PRIMARY_SCALE_ID = _base.PRIMARY_SCALE_ID
FOCAL_ARM = _base.FOCAL_ARM
BASELINE_ARM = _base.BASELINE_ARM
BACKGROUND_ARM = _base.BACKGROUND_ARM
REFERENCE_ARM = _base.REFERENCE_ARM
ARM_ORDER = _base.ARM_ORDER
METRIC_ORDER = _base.METRIC_ORDER
METRIC_SHORT_LABELS = _base.METRIC_SHORT_LABELS

APPENDIX_PAGE_SIZE = 15
APPENDIX_MULTIPLEX_PAGES = 4
PORTABLE_TABLE_ROW_LIMIT = 15
PORTABLE_CHART_ROW_LIMIT = 50
PORTABLE_DATASET_LIMIT = 50
PORTABLE_DATASET_FIELD_LIMIT = 80
PORTABLE_ARTIFACT_BYTE_LIMIT = 3_000_000
EXPECTED_ESTIMATE_ROWS = 600
EXPECTED_CONTRAST_ROWS = 400
EXPECTED_VIEW_ROBUSTNESS_ROWS = 25
EXPECTED_SUPPORT_ROWS = 2 * len(ARM_ORDER)
EXPECTED_SCATTER_ROWS_PER_BACKEND = PRIMARY_N_MEETINGS
POINT_DISPLAY_SUFFIX = " (6 d.p.)"

SUPPORT_EXACT_FIELDS = {
    "mean_raw_score": "mean_raw_score_display",
    "raw_score_sd": "raw_score_sd_display",
    "raw_score_min": "raw_score_min_display",
    "raw_score_p05": "raw_score_p05_display",
    "raw_score_p95": "raw_score_p95_display",
    "raw_score_max": "raw_score_max_display",
}
SCATTER_EXACT_FIELDS = {
    "reference_raw_score": "reference_raw_score_display",
    "chk2_raw_score": "chk2_raw_score_display",
    "raw_error": "raw_error_display",
}

PAPER_STAGE_DISCLOSURE = _base.PAPER_STAGE_DISCLOSURE
REFERENCE_DISCLOSURE = _base.REFERENCE_DISCLOSURE
MEETING_GRAIN_DISCLOSURE = _base.MEETING_GRAIN_DISCLOSURE
IDENTITY_LINE_DISCLOSURE = _base.IDENTITY_LINE_DISCLOSURE
NONPOOLED_DISCLOSURE = _base.NONPOOLED_DISCLOSURE

SentimentClosenessReportError = _base.SentimentClosenessReportError
_binding = _base._binding
_canonical = _base._canonical
_display_arm = _base._display_arm
_intended_binding = _base._intended_binding
_normalize_estimator = _base._normalize_estimator
_package_report = _base._package_report
_report_runtime = _base._report_runtime
_sha256_file = _base._sha256_file
_write_new_json = _base._write_new_json
load_report_model = _base.load_report_model
validate_report_model = _base.validate_report_model

_ESTIMATE_DATASET_RE = re.compile(r"^all_estimates_page_\d{2}$")
_CONTRAST_DATASET_RE = re.compile(r"^all_contrasts_page_\d{2}$")
_ESTIMATE_TABLE_RE = re.compile(r"^all_estimates_page_\d{2}_table$")
_CONTRAST_TABLE_RE = re.compile(r"^all_contrasts_page_\d{2}_table$")
_VIEW_TABLE_RE = re.compile(r"^view_robustness_page_\d{2}_table$")
_COMPACT_ZERO_RE = re.compile(r"^[+-]?(?:0+(?:\.0*)?|\.0+)$")


def _normalized_contract() -> dict[str, Any]:
    contract = copy.deepcopy(_base._normalized_contract())
    contract.update(
        {
            "report_manifest_schema": REPORT_MANIFEST_SCHEMA,
            "report_artifact_contract": REPORT_ARTIFACT_CONTRACT,
            "report_id": REPORT_ID,
            "presentation_revision": {
                "base_renderer": "render_chk3_sentiment_score_closeness_report_v1",
                "appendix_page_size": APPENDIX_PAGE_SIZE,
                "portable_table_row_limit": PORTABLE_TABLE_ROW_LIMIT,
                "portable_chart_row_limit": PORTABLE_CHART_ROW_LIMIT,
                "reference_scale_display": "round-trip float text with exact suffix",
                "support_raw_stat_display": (
                    "round-trip float text with exact suffix for mean, SD, minimum, "
                    "p05, p95, and maximum"
                ),
                "point_display": (
                    "six-decimal estimate/gain text with a nonnumeric 6 d.p. suffix"
                ),
                "scatter_display": (
                    "numeric x/y coordinates plus round-trip exact text companions"
                ),
                "html_acceptance": (
                    "no Showing first; exact 600/400 visible appendix IDs without "
                    "duplicates and with exact display reconciliation; 10 headline "
                    "point cards, 256 scatter rows, 25 view rows, exact reference "
                    "scales, and Distil reference support mean/SD visible"
                ),
            },
        }
    )
    return contract


def _pages(rows: Sequence[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size <= 0:
        raise SentimentClosenessReportError("page size must be positive")
    return [list(rows[index : index + size]) for index in range(0, len(rows), size)]


def _replace_contiguous(
    items: list[dict[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool],
    replacements: Sequence[dict[str, Any]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    indices = [index for index, item in enumerate(items) if predicate(item)]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise SentimentClosenessReportError(
            f"{label} presentation inventory is missing or non-contiguous"
        )
    return [
        *items[: indices[0]],
        *copy.deepcopy(list(replacements)),
        *items[indices[-1] + 1 :],
    ]


def _collect_appendix_rows(
    datasets: Mapping[str, Any],
    *,
    pattern: re.Pattern[str],
    row_id: str,
    expected_rows: int,
    label: str,
) -> list[dict[str, Any]]:
    names = sorted(name for name in datasets if pattern.fullmatch(name))
    if not names:
        raise SentimentClosenessReportError(f"{label} appendix datasets are missing")
    rows: list[dict[str, Any]] = []
    for name in names:
        page = datasets[name]
        if not isinstance(page, list) or any(
            not isinstance(row, Mapping) for row in page
        ):
            raise SentimentClosenessReportError(f"{label} appendix page drift: {name}")
        rows.extend(copy.deepcopy(dict(row)) for row in page)
    ids = [row.get(row_id) for row in rows]
    if (
        len(rows) != expected_rows
        or any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != expected_rows
    ):
        raise SentimentClosenessReportError(
            f"{label} appendix must contain {expected_rows} unique rows"
        )
    return rows


def _repage_appendices(artifact: dict[str, Any]) -> None:
    datasets = artifact["snapshot"]["datasets"]
    tables = artifact["manifest"]["tables"]
    blocks = artifact["manifest"]["blocks"]

    estimates = _collect_appendix_rows(
        datasets,
        pattern=_ESTIMATE_DATASET_RE,
        row_id="estimate_id",
        expected_rows=EXPECTED_ESTIMATE_ROWS,
        label="estimate",
    )
    contrasts = _collect_appendix_rows(
        datasets,
        pattern=_CONTRAST_DATASET_RE,
        row_id="contrast_row_id",
        expected_rows=EXPECTED_CONTRAST_ROWS,
        label="contrast",
    )
    estimates.sort(key=lambda row: str(row["estimate_id"]))
    contrasts.sort(key=lambda row: str(row["contrast_row_id"]))
    estimate_template = next(
        (
            table
            for table in tables
            if _ESTIMATE_TABLE_RE.fullmatch(str(table.get("id")))
        ),
        None,
    )
    contrast_template = next(
        (
            table
            for table in tables
            if _CONTRAST_TABLE_RE.fullmatch(str(table.get("id")))
        ),
        None,
    )
    estimate_block_template = next(
        (
            block
            for block in blocks
            if _ESTIMATE_TABLE_RE.fullmatch(str(block.get("tableId")))
        ),
        None,
    )
    contrast_block_template = next(
        (
            block
            for block in blocks
            if _CONTRAST_TABLE_RE.fullmatch(str(block.get("tableId")))
        ),
        None,
    )
    if any(
        template is None
        for template in (
            estimate_template,
            contrast_template,
            estimate_block_template,
            contrast_block_template,
        )
    ):
        raise SentimentClosenessReportError(
            "appendix table/block templates are missing"
        )

    for name in list(datasets):
        if _ESTIMATE_DATASET_RE.fullmatch(name) or _CONTRAST_DATASET_RE.fullmatch(name):
            del datasets[name]

    def make_family(
        rows: list[dict[str, Any]],
        *,
        prefix: str,
        table_template: Mapping[str, Any],
        block_template: Mapping[str, Any],
        title: str,
        subtitle: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        new_tables = []
        new_blocks = []
        logical_pages = list(enumerate(_pages(rows, APPENDIX_PAGE_SIZE), start=1))
        source_field_count = max(len(row) for row in rows)
        multiplex_pages = min(
            APPENDIX_MULTIPLEX_PAGES,
            PORTABLE_DATASET_FIELD_LIMIT // source_field_count,
        )
        if multiplex_pages <= 0:
            raise SentimentClosenessReportError(
                f"{prefix} rows exceed the portable {PORTABLE_DATASET_FIELD_LIMIT}-field "
                "limit"
            )
        groups: list[list[tuple[int, list[dict[str, Any]]]]] = []
        for page_no, page in logical_pages:
            if (
                not groups
                or len(groups[-1]) >= multiplex_pages
                or len(groups[-1][0][1]) != len(page)
            ):
                groups.append([])
            groups[-1].append((page_no, page))
        template_columns = table_template.get("columns")
        template_sort = table_template.get("defaultSort")
        if not isinstance(template_columns, list) or not isinstance(
            template_sort, Mapping
        ):
            raise SentimentClosenessReportError(
                f"{prefix} appendix column/sort template drift"
            )
        for group_no, group in enumerate(groups, start=1):
            dataset_id = f"{prefix}_multiplex_{group_no:02d}"
            physical_rows: list[dict[str, Any]] = []
            for row_index in range(len(group[0][1])):
                physical: dict[str, Any] = {}
                for page_no, page in group:
                    field_prefix = f"p{page_no:02d}__"
                    physical.update(
                        {
                            f"{field_prefix}{field}": value
                            for field, value in page[row_index].items()
                        }
                    )
                physical_rows.append(physical)
            datasets[dataset_id] = physical_rows
            for page_no, _page in group:
                field_prefix = f"p{page_no:02d}__"
                table_id = f"{prefix}_page_{page_no:02d}_table"
                table = copy.deepcopy(dict(table_template))
                table.update(
                    {
                        "id": table_id,
                        "title": f"{title} — page {page_no}",
                        "subtitle": subtitle,
                        "dataset": dataset_id,
                        "columns": [
                            {
                                **copy.deepcopy(dict(column)),
                                "field": f"{field_prefix}{column['field']}",
                            }
                            for column in template_columns
                        ],
                        "defaultSort": {
                            **copy.deepcopy(dict(template_sort)),
                            "field": f"{field_prefix}{template_sort['field']}",
                        },
                    }
                )
                block = copy.deepcopy(dict(block_template))
                block.update({"id": f"{table_id}_block", "tableId": table_id})
                new_tables.append(table)
                new_blocks.append(block)
        return new_tables, new_blocks

    estimate_tables, estimate_blocks = make_family(
        estimates,
        prefix="all_estimates",
        table_template=estimate_template,
        block_template=estimate_block_template,
        title="Complete estimate audit",
        subtitle="Sealed direct estimates; 15 rows or fewer per page.",
    )
    contrast_tables, contrast_blocks = make_family(
        contrasts,
        prefix="all_contrasts",
        table_template=contrast_template,
        block_template=contrast_block_template,
        title="Complete gain audit",
        subtitle="Sealed paired gains; 15 rows or fewer per page.",
    )
    artifact["manifest"]["tables"] = _replace_contiguous(
        tables,
        lambda table: bool(_ESTIMATE_TABLE_RE.fullmatch(str(table.get("id")))),
        estimate_tables,
        label="estimate appendix tables",
    )
    artifact["manifest"]["tables"] = _replace_contiguous(
        artifact["manifest"]["tables"],
        lambda table: bool(_CONTRAST_TABLE_RE.fullmatch(str(table.get("id")))),
        contrast_tables,
        label="contrast appendix tables",
    )
    artifact["manifest"]["blocks"] = _replace_contiguous(
        blocks,
        lambda block: bool(_ESTIMATE_TABLE_RE.fullmatch(str(block.get("tableId", "")))),
        estimate_blocks,
        label="estimate appendix blocks",
    )
    artifact["manifest"]["blocks"] = _replace_contiguous(
        artifact["manifest"]["blocks"],
        lambda block: bool(_CONTRAST_TABLE_RE.fullmatch(str(block.get("tableId", "")))),
        contrast_blocks,
        label="contrast appendix blocks",
    )


def _split_visible_table(
    artifact: dict[str, Any], *, table_id: str, expected_rows: int
) -> None:
    tables = artifact["manifest"]["tables"]
    blocks = artifact["manifest"]["blocks"]
    datasets = artifact["snapshot"]["datasets"]
    matches = [table for table in tables if table.get("id") == table_id]
    block_matches = [block for block in blocks if block.get("tableId") == table_id]
    if len(matches) != 1 or len(block_matches) != 1:
        raise SentimentClosenessReportError(f"{table_id} table/block identity drift")
    template = matches[0]
    block_template = block_matches[0]
    dataset_id = str(template["dataset"])
    rows = datasets.get(dataset_id)
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise SentimentClosenessReportError(f"{table_id} row coverage drift")
    pages = _pages(rows, PORTABLE_TABLE_ROW_LIMIT)
    new_tables = []
    new_blocks = []
    for page_no, page in enumerate(pages, start=1):
        page_dataset = f"{dataset_id}_page_{page_no:02d}"
        page_table_id = f"{dataset_id}_page_{page_no:02d}_table"
        datasets[page_dataset] = copy.deepcopy(page)
        table = copy.deepcopy(template)
        table.update(
            {
                "id": page_table_id,
                "title": f"{template['title']} — page {page_no} of {len(pages)}",
                "subtitle": (
                    f"{template.get('subtitle', '')} Reader-visible page with at "
                    f"most {PORTABLE_TABLE_ROW_LIMIT} rows."
                ).strip(),
                "dataset": page_dataset,
            }
        )
        block = copy.deepcopy(block_template)
        block.update({"id": f"{page_table_id}_block", "tableId": page_table_id})
        new_tables.append(table)
        new_blocks.append(block)
    del datasets[dataset_id]
    artifact["manifest"]["tables"] = _replace_contiguous(
        tables,
        lambda table: table.get("id") == table_id,
        new_tables,
        label=table_id,
    )
    artifact["manifest"]["blocks"] = _replace_contiguous(
        blocks,
        lambda block: block.get("tableId") == table_id,
        new_blocks,
        label=f"{table_id} blocks",
    )


def _sort_view_robustness(artifact: dict[str, Any]) -> None:
    """Freeze deterministic view paging and its visible default order."""

    rows = artifact["snapshot"]["datasets"].get("view_robustness")
    tables = [
        table
        for table in artifact["manifest"]["tables"]
        if table.get("id") == "view_robustness_table"
    ]
    if (
        not isinstance(rows, list)
        or len(rows) != EXPECTED_VIEW_ROBUSTNESS_ROWS
        or any(not isinstance(row, dict) for row in rows)
        or len(tables) != 1
    ):
        raise SentimentClosenessReportError("view-robustness ordering inventory drift")
    identities = [str(row.get("contrast_row_id", "")) for row in rows]
    if any(not identity for identity in identities) or len(set(identities)) != len(
        identities
    ):
        raise SentimentClosenessReportError("view-robustness ordering identity drift")
    rows.sort(key=lambda row: str(row["contrast_row_id"]))
    tables[0]["defaultSort"] = {
        "field": "contrast_row_id",
        "direction": "asc",
    }


def _split_scatter_charts(artifact: dict[str, Any]) -> None:
    charts = artifact["manifest"]["charts"]
    blocks = artifact["manifest"]["blocks"]
    datasets = artifact["snapshot"]["datasets"]
    new_charts: list[dict[str, Any]] = []
    block_replacements: dict[str, list[dict[str, Any]]] = {}
    for chart in charts:
        dataset_id = str(chart["dataset"])
        rows = datasets.get(dataset_id)
        if not isinstance(rows, list):
            raise SentimentClosenessReportError(
                f"chart dataset is missing or invalid: {dataset_id}"
            )
        if len(rows) <= PORTABLE_CHART_ROW_LIMIT:
            new_charts.append(copy.deepcopy(chart))
            continue
        matching_blocks = [
            block for block in blocks if block.get("chartId") == chart["id"]
        ]
        if len(matching_blocks) != 1:
            raise SentimentClosenessReportError(
                f"chart block identity drift: {chart['id']}"
            )
        template_block = matching_blocks[0]
        ordered_rows = sorted(
            rows,
            key=lambda row: (str(row.get("meeting_date")), str(row.get("meeting_id"))),
        )
        pages = _pages(ordered_rows, PORTABLE_CHART_ROW_LIMIT)
        replacement_blocks = []
        for page_no, page in enumerate(pages, start=1):
            page_dataset = f"{dataset_id}_page_{page_no:02d}"
            page_chart_id = f"{dataset_id}_page_{page_no:02d}_chart"
            datasets[page_dataset] = copy.deepcopy(page)
            page_chart = copy.deepcopy(chart)
            page_chart.update(
                {
                    "id": page_chart_id,
                    "title": f"{chart['title']} — panel {page_no} of {len(pages)}",
                    "subtitle": (
                        f"Pre-2009 shared-complete meetings, panel {page_no} of "
                        f"{len(pages)} (panel N={len(page)}; panels collectively "
                        f"preserve all N={len(rows)} meeting-level pairs)."
                    ),
                    "rationale": (
                        f"Chronologically ordered panel {page_no} of {len(pages)}; "
                        f"the panels collectively preserve all {len(rows)} paired "
                        "observations without pooling replicates or truncating the "
                        "portable semantic fallback."
                    ),
                    "dataset": page_dataset,
                    "maxRows": len(page),
                }
            )
            block = copy.deepcopy(template_block)
            block.update(
                {
                    "id": f"{template_block['id']}_page_{page_no:02d}",
                    "chartId": page_chart_id,
                }
            )
            new_charts.append(page_chart)
            replacement_blocks.append(block)
        block_replacements[str(chart["id"])] = replacement_blocks
        del datasets[dataset_id]

    for chart_id, replacement in block_replacements.items():
        blocks = _replace_contiguous(
            blocks,
            lambda block, chart_id=chart_id: block.get("chartId") == chart_id,
            replacement,
            label=f"{chart_id} blocks",
        )
    artifact["manifest"]["charts"] = new_charts
    artifact["manifest"]["blocks"] = blocks


def _exact_reference_scale_text(artifact: dict[str, Any]) -> None:
    rows = artifact["snapshot"]["datasets"].get("reference_scales")
    if not isinstance(rows, list) or len(rows) != 2:
        raise SentimentClosenessReportError("reference-scale display rows drift")
    for row in rows:
        mean = row.get("reference_mean")
        sd = row.get("reference_sd_ddof1")
        if isinstance(mean, bool) or not isinstance(mean, (int, float)):
            raise SentimentClosenessReportError("reference mean is not numeric")
        if isinstance(sd, bool) or not isinstance(sd, (int, float)) or float(sd) <= 0:
            raise SentimentClosenessReportError("reference SD is not positive")
        row["reference_mean_display"] = f"{float(mean)!r} (exact)"
        row["reference_sd_display"] = f"{float(sd)!r} (exact)"


def _exact_support_diagnostic_text(artifact: dict[str, Any]) -> None:
    """Preserve raw-score support precision in the portable table fallback."""

    datasets = artifact["snapshot"]["datasets"]
    rows = datasets.get("support_diagnostics")
    if not isinstance(rows, list) or len(rows) != EXPECTED_SUPPORT_ROWS:
        raise SentimentClosenessReportError("support-diagnostic display rows drift")
    for row in rows:
        if not isinstance(row, dict):
            raise SentimentClosenessReportError(
                "support-diagnostic display row is invalid"
            )
        for source_field, display_field in SUPPORT_EXACT_FIELDS.items():
            value = row.get(source_field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise SentimentClosenessReportError(
                    f"support diagnostic {source_field} is not finite numeric"
                )
            row[display_field] = f"{float(value)!r} (exact)"

    matching_tables = [
        table
        for table in artifact["manifest"]["tables"]
        if table.get("id") == "support_diagnostics_table"
    ]
    if len(matching_tables) != 1:
        raise SentimentClosenessReportError("support-diagnostics table identity drift")
    columns = matching_tables[0].get("columns")
    if not isinstance(columns, list):
        raise SentimentClosenessReportError(
            "support-diagnostics table column inventory drift"
        )
    replaced: set[str] = set()
    for column in columns:
        source_field = str(column.get("field"))
        display_field = SUPPORT_EXACT_FIELDS.get(source_field)
        if display_field is None:
            continue
        column["field"] = display_field
        column["type"] = "text"
        column.pop("format", None)
        replaced.add(source_field)
    if replaced != set(SUPPORT_EXACT_FIELDS):
        raise SentimentClosenessReportError(
            "support-diagnostics exact-text column binding drift"
        )


def _reported_point_text(value: Any) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SentimentClosenessReportError(
            "reported point value is not finite numeric"
        )
    return f"{float(value):+.6f}{POINT_DISPLAY_SUFFIX}"


def _exact_point_displays(artifact: dict[str, Any]) -> None:
    """Make every reader-facing estimate/gain point resistant to JS coercion."""

    for rows in artifact["snapshot"]["datasets"].values():
        if not isinstance(rows, list):
            raise SentimentClosenessReportError("report dataset inventory drift")
        for row in rows:
            if not isinstance(row, dict):
                raise SentimentClosenessReportError("report dataset row type drift")
            if "estimate_id" in row:
                row["estimate_display"] = _reported_point_text(row.get("estimate"))
            if "contrast_row_id" in row:
                row["gain_display"] = _reported_point_text(row.get("gain"))

    headline_rows = artifact["snapshot"]["datasets"].get("headline")
    if not isinstance(headline_rows, list) or len(headline_rows) != 1:
        raise SentimentClosenessReportError("headline point display row drift")
    headline = headline_rows[0]
    if not isinstance(headline, dict):
        raise SentimentClosenessReportError("headline point display type drift")
    for metric in METRIC_ORDER:
        for kind in ("estimate", "gain"):
            field = f"{metric}_{kind}"
            value = headline.get(field)
            if not isinstance(value, str):
                raise SentimentClosenessReportError(
                    f"headline point display field drift: {field}"
                )
            numeric_text = value.removesuffix(POINT_DISPLAY_SUFFIX)
            try:
                numeric_value = float(numeric_text)
            except ValueError as exc:
                raise SentimentClosenessReportError(
                    f"headline point display is not numeric before suffix: {field}"
                ) from exc
            headline[field] = _reported_point_text(numeric_value)


def _exact_scatter_text(artifact: dict[str, Any]) -> None:
    """Add exact tooltip companions while retaining numeric scatter coordinates."""

    datasets = artifact["snapshot"]["datasets"]
    for dataset_id in ("distil_raw_scatter", "finbert_raw_scatter"):
        rows = datasets.get(dataset_id)
        if not isinstance(rows, list) or len(rows) != EXPECTED_SCATTER_ROWS_PER_BACKEND:
            raise SentimentClosenessReportError(
                f"scatter exact-text row coverage drift: {dataset_id}"
            )
        for row in rows:
            if not isinstance(row, dict):
                raise SentimentClosenessReportError(
                    f"scatter exact-text row type drift: {dataset_id}"
                )
            for source_field, display_field in SCATTER_EXACT_FIELDS.items():
                value = row.get(source_field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise SentimentClosenessReportError(
                        f"scatter {source_field} is not finite numeric"
                    )
                row[display_field] = f"{float(value)!r} (exact)"

    charts = artifact["manifest"]["charts"]
    matches = [
        chart
        for chart in charts
        if chart.get("id") in {"distil_raw_scatter_chart", "finbert_raw_scatter_chart"}
    ]
    if len(matches) != 2:
        raise SentimentClosenessReportError("scatter chart identity drift")
    for chart in matches:
        encodings = chart.get("encodings")
        if not isinstance(encodings, dict) or not isinstance(
            encodings.get("tooltip"), list
        ):
            raise SentimentClosenessReportError("scatter tooltip inventory drift")
        tooltip = encodings["tooltip"]
        replacements = set()
        for encoding in tooltip:
            source_field = str(encoding.get("field", ""))
            display_field = SCATTER_EXACT_FIELDS.get(source_field)
            if display_field is None:
                source_field = next(
                    (
                        source
                        for source, display in SCATTER_EXACT_FIELDS.items()
                        if display == encoding.get("field")
                    ),
                    source_field,
                )
                display_field = SCATTER_EXACT_FIELDS.get(source_field)
            if display_field is None:
                continue
            encoding["field"] = display_field
            encoding["type"] = "nominal"
            label = str(encoding.get("label", source_field))
            encoding["label"] = (
                label if label.endswith(" (exact)") else f"{label} (exact)"
            )
            replacements.add(source_field)
        if replacements != set(SCATTER_EXACT_FIELDS):
            raise SentimentClosenessReportError(
                f"scatter exact tooltip binding drift: {chart.get('id')}"
            )


def _appendix_ids_by_table(
    artifact: Mapping[str, Any], *, prefix: str, row_id: str, expected: int
) -> dict[str, list[str]]:
    datasets = artifact["snapshot"]["datasets"]
    pattern = re.compile(rf"^{re.escape(prefix)}_page_\d{{2}}_table$")
    tables = sorted(
        (
            table
            for table in artifact["manifest"]["tables"]
            if pattern.fullmatch(str(table.get("id")))
        ),
        key=lambda table: str(table["id"]),
    )
    ids_by_table: dict[str, list[str]] = {}
    for table in tables:
        rows = datasets.get(str(table.get("dataset")))
        columns = table.get("columns")
        if not isinstance(rows, list) or not isinstance(columns, list):
            raise SentimentClosenessReportError(
                f"{prefix} v2 appendix table dataset/column drift"
            )
        id_fields = [
            str(column.get("field"))
            for column in columns
            if str(column.get("field", "")).endswith(f"__{row_id}")
        ]
        if len(id_fields) != 1:
            raise SentimentClosenessReportError(
                f"{prefix} v2 appendix ID column drift: {table.get('id')}"
            )
        ids_by_table[str(table["id"])] = [
            str(row.get(id_fields[0], "")) for row in rows
        ]
    ids = [value for values in ids_by_table.values() for value in values]
    if (
        len(ids) != expected
        or any(not value for value in ids)
        or len(set(ids)) != expected
    ):
        raise SentimentClosenessReportError(f"{prefix} v2 appendix row identity drift")
    return ids_by_table


def _appendix_ids_from_v2(
    artifact: Mapping[str, Any], *, prefix: str, row_id: str, expected: int
) -> list[str]:
    return [
        value
        for values in _appendix_ids_by_table(
            artifact, prefix=prefix, row_id=row_id, expected=expected
        ).values()
        for value in values
    ]


def _appendix_displays_by_table(
    artifact: Mapping[str, Any],
    *,
    prefix: str,
    row_id: str,
    numeric_field: str,
    display_field: str,
    expected: int,
) -> dict[str, dict[str, str]]:
    datasets = artifact["snapshot"]["datasets"]
    pattern = re.compile(rf"^{re.escape(prefix)}_page_\d{{2}}_table$")
    tables = sorted(
        (
            table
            for table in artifact["manifest"]["tables"]
            if pattern.fullmatch(str(table.get("id")))
        ),
        key=lambda table: str(table["id"]),
    )
    values_by_table: dict[str, dict[str, str]] = {}
    for table in tables:
        columns = table.get("columns")
        rows = datasets.get(str(table.get("dataset")))
        if not isinstance(columns, list) or not isinstance(rows, list):
            raise SentimentClosenessReportError(
                f"{prefix} display table dataset/column drift"
            )

        def one_field(suffix: str) -> str:
            matches = [
                str(column.get("field"))
                for column in columns
                if str(column.get("field", "")).endswith(f"__{suffix}")
            ]
            if len(matches) != 1:
                raise SentimentClosenessReportError(
                    f"{prefix} display column drift: {table.get('id')}/{suffix}"
                )
            return matches[0]

        id_key = one_field(row_id)
        display_key = one_field(display_field)
        field_prefix = id_key[: -len(row_id)]
        numeric_key = f"{field_prefix}{numeric_field}"
        expected_values: dict[str, str] = {}
        for row in rows:
            if numeric_key not in row:
                raise SentimentClosenessReportError(
                    f"{prefix} numeric source field drift: {table.get('id')}"
                )
            row_identity = str(row.get(id_key, ""))
            if not row_identity or row_identity in expected_values:
                raise SentimentClosenessReportError(
                    f"{prefix} display row identity drift: {table.get('id')}"
                )
            display = row.get(display_key)
            expected_display = _reported_point_text(row.get(numeric_key))
            if display != expected_display:
                raise SentimentClosenessReportError(
                    f"{prefix} coercion-safe display drift: {row_identity}"
                )
            expected_values[row_identity] = str(display)
        values_by_table[str(table["id"])] = expected_values
    total = sum(len(values) for values in values_by_table.values())
    identities = {
        identity for values in values_by_table.values() for identity in values
    }
    if total != expected or len(identities) != expected:
        raise SentimentClosenessReportError(
            f"{prefix} coercion-safe display coverage drift"
        )
    return values_by_table


def _appendix_records_by_id(
    artifact: Mapping[str, Any],
    *,
    prefix: str,
    row_id: str,
    expected: int,
) -> dict[str, dict[str, Any]]:
    """Reconstruct logical rows from horizontally multiplexed appendix datasets."""

    datasets = artifact["snapshot"]["datasets"]
    pattern = re.compile(rf"^{re.escape(prefix)}_page_\d{{2}}_table$")
    tables = sorted(
        (
            table
            for table in artifact["manifest"]["tables"]
            if pattern.fullmatch(str(table.get("id")))
        ),
        key=lambda table: str(table["id"]),
    )
    result: dict[str, dict[str, Any]] = {}
    for table in tables:
        columns = table.get("columns")
        rows = datasets.get(str(table.get("dataset")))
        if not isinstance(columns, list) or not isinstance(rows, list):
            raise SentimentClosenessReportError(
                f"{prefix} logical appendix inventory drift"
            )
        id_fields = [
            str(column.get("field"))
            for column in columns
            if str(column.get("field", "")).endswith(f"__{row_id}")
        ]
        if len(id_fields) != 1:
            raise SentimentClosenessReportError(
                f"{prefix} logical appendix ID binding drift: {table.get('id')}"
            )
        id_key = id_fields[0]
        field_prefix = id_key[: -len(row_id)]
        for row in rows:
            logical = {
                field[len(field_prefix) :]: copy.deepcopy(value)
                for field, value in row.items()
                if field.startswith(field_prefix)
            }
            identity = str(logical.get(row_id, ""))
            if not identity or identity in result:
                raise SentimentClosenessReportError(
                    f"{prefix} logical appendix row identity drift"
                )
            result[identity] = logical
    if len(result) != expected:
        raise SentimentClosenessReportError(
            f"{prefix} logical appendix row coverage drift"
        )
    return result


def _view_ids_by_table(artifact: Mapping[str, Any]) -> dict[str, list[str]]:
    datasets = artifact["snapshot"]["datasets"]
    tables = sorted(
        (
            table
            for table in artifact["manifest"]["tables"]
            if _VIEW_TABLE_RE.fullmatch(str(table.get("id")))
        ),
        key=lambda table: str(table["id"]),
    )
    expected_table_ids = {
        "view_robustness_page_01_table",
        "view_robustness_page_02_table",
    }
    if {str(table.get("id")) for table in tables} != expected_table_ids:
        raise SentimentClosenessReportError("view-robustness table inventory drift")
    authoritative = _appendix_records_by_id(
        artifact,
        prefix="all_contrasts",
        row_id="contrast_row_id",
        expected=EXPECTED_CONTRAST_ROWS,
    )
    result: dict[str, list[str]] = {}
    for table in tables:
        rows = datasets.get(str(table.get("dataset")))
        if not isinstance(rows, list) or any(
            not isinstance(row, Mapping) for row in rows
        ):
            raise SentimentClosenessReportError("view-robustness dataset drift")
        if table.get("defaultSort") != {
            "field": "contrast_row_id",
            "direction": "asc",
        }:
            raise SentimentClosenessReportError(
                "view-robustness default-sort binding drift"
            )
        identities = [str(row.get("contrast_row_id", "")) for row in rows]
        for row, identity in zip(rows, identities, strict=True):
            source = authoritative.get(identity)
            if (
                source is None
                or row.get("gain") != source.get("gain")
                or row.get("gain_display") != source.get("gain_display")
            ):
                raise SentimentClosenessReportError(
                    f"view-robustness authoritative gain drift: {identity}"
                )
        result[str(table["id"])] = identities
    identities = [identity for rows in result.values() for identity in rows]
    if (
        [len(result[table_id]) for table_id in sorted(result)] != [15, 10]
        or len(identities) != EXPECTED_VIEW_ROBUSTNESS_ROWS
        or any(not identity for identity in identities)
        or len(set(identities)) != EXPECTED_VIEW_ROBUSTNESS_ROWS
        or identities != sorted(identities)
    ):
        raise SentimentClosenessReportError("view-robustness row identity drift")
    return result


def _headline_point_contract(artifact: Mapping[str, Any]) -> dict[str, str]:
    rows = artifact["snapshot"]["datasets"].get("headline")
    cards = artifact["manifest"].get("cards")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(cards, list):
        raise SentimentClosenessReportError("headline point contract inventory drift")
    headline = rows[0]
    if not isinstance(headline, Mapping):
        raise SentimentClosenessReportError("headline point contract row drift")
    card_ids = [str(card.get("id", "")) for card in cards]
    cards_by_id = {str(card.get("id")): card for card in cards}
    expected_card_ids = {
        "headline_n_card",
        "headline_bootstrap_card",
        *(
            f"headline_{metric}_{suffix}_card"
            for metric in METRIC_ORDER
            for suffix in ("direct", "gain")
        ),
    }
    if (
        any(not card_id for card_id in card_ids)
        or len(card_ids) != len(set(card_ids))
        or set(card_ids) != expected_card_ids
    ):
        raise SentimentClosenessReportError("headline card ID inventory drift")
    estimates = _appendix_records_by_id(
        artifact,
        prefix="all_estimates",
        row_id="estimate_id",
        expected=EXPECTED_ESTIMATE_ROWS,
    )
    contrasts = _appendix_records_by_id(
        artifact,
        prefix="all_contrasts",
        row_id="contrast_row_id",
        expected=EXPECTED_CONTRAST_ROWS,
    )
    expected: dict[str, str] = {}
    for metric in METRIC_ORDER:
        for kind, card_suffix in (("estimate", "direct"), ("gain", "gain")):
            field = f"{metric}_{kind}"
            card_id = f"headline_{metric}_{card_suffix}_card"
            value = headline.get(field)
            card = cards_by_id.get(card_id)
            metrics = card.get("metrics") if isinstance(card, Mapping) else None
            if (
                not isinstance(value, str)
                or not value.endswith(POINT_DISPLAY_SUFFIX)
                or not isinstance(metrics, list)
                or not metrics
                or metrics[0].get("field") != field
            ):
                raise SentimentClosenessReportError(
                    f"headline coercion-safe point binding drift: {card_id}"
                )
            source_rows = (
                estimates.values() if kind == "estimate" else contrasts.values()
            )
            authoritative_rows = [
                row
                for row in source_rows
                if row.get("backend_id") == PRIMARY_BACKEND_ID
                and row.get("view") == PRIMARY_VIEW_ID
                and row.get("score_policy") == PRIMARY_SCORE_POLICY
                and row.get("scale") == PRIMARY_SCALE_ID
                and row.get("metric_id") == metric
                and (
                    row.get("artifact_arm") == FOCAL_ARM
                    if kind == "estimate"
                    else row.get("baseline_artifact_arm") == BASELINE_ARM
                )
            ]
            authoritative_field = (
                "estimate_display" if kind == "estimate" else "gain_display"
            )
            if len(authoritative_rows) != 1 or value != authoritative_rows[0].get(
                authoritative_field
            ):
                raise SentimentClosenessReportError(
                    f"headline authoritative point drift: {card_id}"
                )
            try:
                float(value)
            except ValueError:
                pass
            else:
                raise SentimentClosenessReportError(
                    f"headline point remains numeric-coercible: {card_id}"
                )
            expected[card_id] = value
    if len(expected) != 2 * len(METRIC_ORDER):
        raise SentimentClosenessReportError("headline point-card coverage drift")
    return expected


def _scatter_contract(artifact: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    datasets = artifact["snapshot"]["datasets"]
    charts = artifact["manifest"].get("charts")
    if not isinstance(charts, list):
        raise SentimentClosenessReportError("scatter chart contract inventory drift")
    expected_chart_sizes = {
        f"{backend}_raw_scatter_page_{page_no:02d}_chart": size
        for backend in ("distil", "finbert")
        for page_no, size in enumerate((50, 50, 28), start=1)
    }
    chart_ids = [str(chart.get("id", "")) for chart in charts]
    if (
        any(not chart_id for chart_id in chart_ids)
        or len(chart_ids) != len(set(chart_ids))
        or set(chart_ids) != set(expected_chart_sizes)
    ):
        raise SentimentClosenessReportError("scatter chart ID inventory drift")
    expected_by_chart: dict[str, dict[str, str]] = {}
    backend_counts = {"distil": 0, "finbert": 0}
    backend_identities = {"distil": set(), "finbert": set()}
    for chart in charts:
        chart_id = str(chart.get("id", ""))
        backend = next(
            (
                prefix
                for prefix in ("distil", "finbert")
                if chart_id.startswith(f"{prefix}_raw_scatter_page_")
            ),
            None,
        )
        if backend is None:
            continue
        encodings = chart.get("encodings")
        if not isinstance(encodings, Mapping):
            raise SentimentClosenessReportError(
                f"scatter encoding contract drift: {chart_id}"
            )
        x = encodings.get("x")
        y = encodings.get("y")
        tooltip = encodings.get("tooltip")
        if (
            not isinstance(x, Mapping)
            or x.get("field") != "reference_raw_score"
            or x.get("type") != "quantitative"
            or not isinstance(y, Mapping)
            or y.get("field") != "chk2_raw_score"
            or y.get("type") != "quantitative"
            or not isinstance(tooltip, list)
            or not set(SCATTER_EXACT_FIELDS.values())
            <= {encoding.get("field") for encoding in tooltip}
        ):
            raise SentimentClosenessReportError(
                f"scatter numeric/exact binding drift: {chart_id}"
            )
        rows = datasets.get(str(chart.get("dataset")))
        expected_dataset = chart_id.removesuffix("_chart")
        if (
            chart.get("dataset") != expected_dataset
            or not isinstance(rows, list)
            or len(rows) != expected_chart_sizes[chart_id]
        ):
            raise SentimentClosenessReportError(f"scatter dataset drift: {chart_id}")
        chart_values: dict[str, str] = {}
        for row in rows:
            point_label = str(row.get("point_label", ""))
            if not point_label or point_label in chart_values:
                raise SentimentClosenessReportError(
                    f"scatter point identity drift: {chart_id}"
                )
            display_values = []
            for source_field, display_field in SCATTER_EXACT_FIELDS.items():
                value = row.get(source_field)
                expected_display = (
                    f"{float(value)!r} (exact)"
                    if not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    else None
                )
                if (
                    expected_display is None
                    or row.get(display_field) != expected_display
                ):
                    raise SentimentClosenessReportError(
                        f"scatter exact display drift: {chart_id}/{point_label}"
                    )
                display_values.append(expected_display)
            chart_values[point_label] = _canonical(display_values)
        expected_by_chart[chart_id] = chart_values
        backend_counts[backend] += len(rows)
        backend_identities[backend].update(chart_values)
    if any(
        backend_counts[backend] != EXPECTED_SCATTER_ROWS_PER_BACKEND
        or len(backend_identities[backend]) != EXPECTED_SCATTER_ROWS_PER_BACKEND
        for backend in backend_counts
    ):
        raise SentimentClosenessReportError("scatter exact display coverage drift")
    return expected_by_chart


def _manifest_id_inventory(artifact: Mapping[str, Any]) -> dict[str, int]:
    manifest = artifact.get("manifest")
    if not isinstance(manifest, Mapping):
        raise SentimentClosenessReportError("manifest ID inventory is missing")
    result: dict[str, int] = {}
    for collection in ("cards", "charts", "tables", "blocks", "sources"):
        items = manifest.get(collection)
        if not isinstance(items, list) or any(
            not isinstance(item, Mapping) for item in items
        ):
            raise SentimentClosenessReportError(
                f"manifest {collection} ID inventory drift"
            )
        identities = [str(item.get("id", "")) for item in items]
        if any(not identity for identity in identities) or len(identities) != len(
            set(identities)
        ):
            raise SentimentClosenessReportError(
                f"manifest {collection} IDs are empty or duplicated"
            )
        result[collection] = len(identities)
    return result


def _validate_artifact_fallback_contract(artifact: Mapping[str, Any]) -> dict[str, Any]:
    manifest = artifact.get("manifest")
    snapshot = artifact.get("snapshot")
    if not isinstance(manifest, Mapping) or not isinstance(snapshot, Mapping):
        raise SentimentClosenessReportError("v2 artifact manifest/snapshot is missing")
    datasets = snapshot.get("datasets")
    tables = manifest.get("tables")
    charts = manifest.get("charts")
    if (
        not isinstance(datasets, Mapping)
        or not isinstance(tables, list)
        or not isinstance(charts, list)
    ):
        raise SentimentClosenessReportError("v2 artifact inventory is invalid")
    manifest_id_counts = _manifest_id_inventory(artifact)
    if len(datasets) > PORTABLE_DATASET_LIMIT:
        raise SentimentClosenessReportError(
            f"v2 artifact has {len(datasets)} datasets; portable maximum is "
            f"{PORTABLE_DATASET_LIMIT}"
        )
    max_dataset_fields = max(
        (len(row) for rows in datasets.values() for row in rows), default=0
    )
    if max_dataset_fields > PORTABLE_DATASET_FIELD_LIMIT:
        raise SentimentClosenessReportError(
            f"v2 artifact dataset row has {max_dataset_fields} fields; portable "
            f"maximum is {PORTABLE_DATASET_FIELD_LIMIT}"
        )
    for table in tables:
        rows = datasets.get(str(table.get("dataset")))
        if not isinstance(rows, list) or len(rows) > PORTABLE_TABLE_ROW_LIMIT:
            raise SentimentClosenessReportError(
                f"portable table fallback would truncate: {table.get('id')}"
            )
    for chart in charts:
        rows = datasets.get(str(chart.get("dataset")))
        if not isinstance(rows, list) or len(rows) > PORTABLE_CHART_ROW_LIMIT:
            raise SentimentClosenessReportError(
                f"portable chart fallback would truncate: {chart.get('id')}"
            )
    estimates = _appendix_ids_from_v2(
        artifact,
        prefix="all_estimates",
        row_id="estimate_id",
        expected=EXPECTED_ESTIMATE_ROWS,
    )
    contrasts = _appendix_ids_from_v2(
        artifact,
        prefix="all_contrasts",
        row_id="contrast_row_id",
        expected=EXPECTED_CONTRAST_ROWS,
    )
    estimate_displays = _appendix_displays_by_table(
        artifact,
        prefix="all_estimates",
        row_id="estimate_id",
        numeric_field="estimate",
        display_field="estimate_display",
        expected=EXPECTED_ESTIMATE_ROWS,
    )
    contrast_displays = _appendix_displays_by_table(
        artifact,
        prefix="all_contrasts",
        row_id="contrast_row_id",
        numeric_field="gain",
        display_field="gain_display",
        expected=EXPECTED_CONTRAST_ROWS,
    )
    headline_points = _headline_point_contract(artifact)
    scatter_points = _scatter_contract(artifact)
    view_ids = _view_ids_by_table(artifact)
    reference_rows = datasets.get("reference_scales")
    if not isinstance(reference_rows, list) or len(reference_rows) != 2:
        raise SentimentClosenessReportError("reference exact-text contract drift")
    for row in reference_rows:
        if not isinstance(row, Mapping):
            raise SentimentClosenessReportError("reference exact-text row type drift")
        expected_mean = f"{float(row.get('reference_mean'))!r} (exact)"
        expected_sd = f"{float(row.get('reference_sd_ddof1'))!r} (exact)"
        if (
            row.get("reference_mean_display") != expected_mean
            or row.get("reference_sd_display") != expected_sd
        ):
            raise SentimentClosenessReportError("reference exact-text contract drift")
    support_rows = datasets.get("support_diagnostics")
    if not isinstance(support_rows, list) or len(support_rows) != EXPECTED_SUPPORT_ROWS:
        raise SentimentClosenessReportError("support exact-text row coverage drift")
    support_keys = [
        (row.get("backend_id"), row.get("artifact_arm"))
        for row in support_rows
        if isinstance(row, Mapping)
    ]
    if len(support_keys) != EXPECTED_SUPPORT_ROWS or len(set(support_keys)) != len(
        support_keys
    ):
        raise SentimentClosenessReportError("support exact-text row identity drift")
    for row in support_rows:
        if not isinstance(row, Mapping):
            raise SentimentClosenessReportError("support exact-text row type drift")
        for source_field, display_field in SUPPORT_EXACT_FIELDS.items():
            value = row.get(source_field)
            expected_display = (
                f"{float(value)!r} (exact)"
                if not isinstance(value, bool) and isinstance(value, (int, float))
                else None
            )
            if (
                expected_display is None
                or not math.isfinite(float(value))
                or row.get(display_field) != expected_display
            ):
                raise SentimentClosenessReportError(
                    f"support exact-text value drift: {source_field}"
                )
    support_tables = [
        table for table in tables if table.get("id") == "support_diagnostics_table"
    ]
    if len(support_tables) != 1 or not isinstance(
        support_tables[0].get("columns"), list
    ):
        raise SentimentClosenessReportError("support exact-text table identity drift")
    support_columns = support_tables[0]["columns"]
    for source_field, display_field in SUPPORT_EXACT_FIELDS.items():
        display_columns = [
            column for column in support_columns if column.get("field") == display_field
        ]
        if (
            len(display_columns) != 1
            or display_columns[0].get("type") != "text"
            or "format" in display_columns[0]
            or any(column.get("field") == source_field for column in support_columns)
        ):
            raise SentimentClosenessReportError(
                f"support exact-text table binding drift: {source_field}"
            )
    artifact_bytes = len(_canonical(artifact).encode("utf-8"))
    if artifact_bytes > PORTABLE_ARTIFACT_BYTE_LIMIT:
        raise SentimentClosenessReportError(
            f"v2 canonical artifact has {artifact_bytes} bytes; portable maximum is "
            f"{PORTABLE_ARTIFACT_BYTE_LIMIT}"
        )
    return {
        "estimate_rows": len(estimates),
        "estimate_display_rows": sum(len(rows) for rows in estimate_displays.values()),
        "contrast_rows": len(contrasts),
        "contrast_display_rows": sum(len(rows) for rows in contrast_displays.values()),
        "estimate_pages": len(estimates) // APPENDIX_PAGE_SIZE,
        "contrast_pages": (len(contrasts) + APPENDIX_PAGE_SIZE - 1)
        // APPENDIX_PAGE_SIZE,
        "reference_scale_rows": len(reference_rows),
        "support_rows": len(support_rows),
        "support_raw_exact_fields": len(SUPPORT_EXACT_FIELDS),
        "headline_point_cards": len(headline_points),
        "scatter_rows": sum(len(rows) for rows in scatter_points.values()),
        "scatter_exact_fields": len(SCATTER_EXACT_FIELDS),
        "view_robustness_rows": sum(len(rows) for rows in view_ids.values()),
        "manifest_id_counts": manifest_id_counts,
        "dataset_count": len(datasets),
        "dataset_limit": PORTABLE_DATASET_LIMIT,
        "max_dataset_fields": max_dataset_fields,
        "dataset_field_limit": PORTABLE_DATASET_FIELD_LIMIT,
        "canonical_artifact_bytes": artifact_bytes,
        "artifact_byte_limit": PORTABLE_ARTIFACT_BYTE_LIMIT,
        "table_row_limit": PORTABLE_TABLE_ROW_LIMIT,
        "chart_row_limit": PORTABLE_CHART_ROW_LIMIT,
    }


def build_canonical_artifact(model: Mapping[str, Any]) -> dict[str, Any]:
    """Build v2 solely by adapting the validated v1 canonical artifact."""

    artifact = _base.build_canonical_artifact(model)
    artifact["manifest"]["description"] = (
        f"{artifact['manifest']['description']} Presentation revision v2 preserves "
        "exact reference-scale and raw support-statistic text plus complete portable "
        "fallback rows."
    )
    _exact_reference_scale_text(artifact)
    _exact_support_diagnostic_text(artifact)
    _exact_point_displays(artifact)
    _exact_scatter_text(artifact)
    _repage_appendices(artifact)
    _sort_view_robustness(artifact)
    _split_visible_table(
        artifact,
        table_id="view_robustness_table",
        expected_rows=EXPECTED_VIEW_ROBUSTNESS_ROWS,
    )
    _split_scatter_charts(artifact)
    _validate_artifact_fallback_contract(artifact)
    return artifact


class _VisibleTableParser(HTMLParser):
    """Collect cells from semantic-fallback tables, excluding embedded JSON."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._section_ids: list[str | None] = []
        self._figure_ids: list[str | None] = []
        self._card_ids: list[str | None] = []
        self._table_id: str | None = None
        self._in_tbody = False
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._card_value: list[str] | None = None
        self.rows: dict[str, list[list[str]]] = {}
        self.card_values: dict[str, list[str]] = {}
        self.table_hosts: dict[str, int] = {}
        self.chart_hosts: dict[str, int] = {}
        self.card_hosts: dict[str, int] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "section":
            table_id = attributes.get("data-table-id")
            self._section_ids.append(table_id)
            if table_id:
                self.table_hosts[table_id] = self.table_hosts.get(table_id, 0) + 1
        elif tag == "figure":
            chart_id = attributes.get("data-chart-id")
            self._figure_ids.append(chart_id)
            if chart_id:
                self.chart_hosts[chart_id] = self.chart_hosts.get(chart_id, 0) + 1
        elif tag == "article":
            card_id = attributes.get("data-card-id")
            self._card_ids.append(card_id)
            if card_id:
                self.card_hosts[card_id] = self.card_hosts.get(card_id, 0) + 1
        elif tag == "table":
            table_id = next(
                (value for value in reversed(self._section_ids) if value), None
            )
            if table_id is None:
                chart_id = next(
                    (value for value in reversed(self._figure_ids) if value), None
                )
                table_id = f"chart:{chart_id}" if chart_id else None
            self._table_id = table_id
        elif tag == "tbody" and self._table_id:
            self._in_tbody = True
        elif tag == "tr" and self._table_id and self._in_tbody:
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell = []
        elif (
            tag == "span"
            and "portable-source-value-text" in str(attributes.get("class", "")).split()
        ):
            if next((value for value in reversed(self._card_ids) if value), None):
                self._card_value = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)
        if self._card_value is not None:
            self._card_value.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._cell is not None and self._row is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "span" and self._card_value is not None:
            card_id = next((value for value in reversed(self._card_ids) if value), None)
            if card_id:
                self.card_values.setdefault(card_id, []).append(
                    "".join(self._card_value).strip()
                )
            self._card_value = None
        elif tag == "tr" and self._row is not None and self._table_id:
            self.rows.setdefault(self._table_id, []).append(self._row)
            self._row = None
        elif tag == "tbody" and self._table_id:
            self._in_tbody = False
        elif tag == "table":
            self._table_id = None
            self._in_tbody = False
            self._row = None
            self._cell = None
        elif tag == "section" and self._section_ids:
            self._section_ids.pop()
        elif tag == "figure" and self._figure_ids:
            self._figure_ids.pop()
        elif tag == "article" and self._card_ids:
            self._card_ids.pop()


def _visible_ids(
    parsed: Mapping[str, list[list[str]]], *, pattern: re.Pattern[str]
) -> list[str]:
    table_ids = sorted(table_id for table_id in parsed if pattern.fullmatch(table_id))
    ids = []
    for table_id in table_ids:
        rows = parsed[table_id]
        if len(rows) > PORTABLE_TABLE_ROW_LIMIT or any(not row for row in rows):
            raise SentimentClosenessReportError(
                f"visible table row contract drift: {table_id}"
            )
        ids.extend(row[0] for row in rows)
    return ids


def _table_fields(artifact: Mapping[str, Any], table_id: str) -> list[str]:
    matches = [
        table for table in artifact["manifest"]["tables"] if table.get("id") == table_id
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("columns"), list):
        raise SentimentClosenessReportError(
            f"packaged HTML table column identity drift: {table_id}"
        )
    fields = [str(column.get("field", "")) for column in matches[0]["columns"]]
    if any(not field for field in fields) or len(set(fields)) != len(fields):
        raise SentimentClosenessReportError(
            f"packaged HTML table column field drift: {table_id}"
        )
    return fields


def _chart_fields(chart: Mapping[str, Any]) -> list[str]:
    encodings = chart.get("encodings")
    if not isinstance(encodings, Mapping):
        raise SentimentClosenessReportError(
            f"packaged HTML chart encoding drift: {chart.get('id')}"
        )
    fields: list[str] = []
    for role in ("x", "y", "color", "size", "facet", "label"):
        encoding = encodings.get(role)
        if not isinstance(encoding, Mapping):
            continue
        candidates = [encoding.get("field")]
        extra = encoding.get("fields")
        if isinstance(extra, list):
            candidates.extend(extra)
        for candidate in candidates:
            if isinstance(candidate, str) and candidate and candidate not in fields:
                fields.append(candidate)
    tooltip = encodings.get("tooltip")
    if isinstance(tooltip, list):
        for encoding in tooltip:
            field = encoding.get("field") if isinstance(encoding, Mapping) else None
            if isinstance(field, str) and field and field not in fields:
                fields.append(field)
    if not fields:
        raise SentimentClosenessReportError(
            f"packaged HTML chart columns are empty: {chart.get('id')}"
        )
    return fields


def _validate_visible_appendix_displays(
    *,
    parser: _VisibleTableParser,
    artifact: Mapping[str, Any],
    expected_by_table: Mapping[str, Mapping[str, str]],
    row_id: str,
    display_field: str,
    label: str,
) -> int:
    total = 0
    for table_id, expected in expected_by_table.items():
        fields = _table_fields(artifact, table_id)
        id_fields = [field for field in fields if field.endswith(f"__{row_id}")]
        display_fields = [
            field for field in fields if field.endswith(f"__{display_field}")
        ]
        if len(id_fields) != 1 or len(display_fields) != 1:
            raise SentimentClosenessReportError(
                f"packaged HTML {label} display column drift: {table_id}"
            )
        id_index = fields.index(id_fields[0])
        display_index = fields.index(display_fields[0])
        visible_rows = parser.rows.get(table_id, [])
        if len(visible_rows) != len(expected) or any(
            len(row) != len(fields) for row in visible_rows
        ):
            raise SentimentClosenessReportError(
                f"packaged HTML {label} display row coverage drift: {table_id}"
            )
        visible = {
            row[id_index]: row[display_index] for row in visible_rows if row[id_index]
        }
        if len(visible) != len(expected) or visible != dict(expected):
            raise SentimentClosenessReportError(
                f"packaged HTML lost exact {label} display values: {table_id}"
            )
        total += len(visible)
    return total


def _validate_visible_view_ids(
    parser: _VisibleTableParser, artifact: Mapping[str, Any]
) -> int:
    expected_by_table = _view_ids_by_table(artifact)
    all_visible: list[str] = []
    for table_id, expected in expected_by_table.items():
        fields = _table_fields(artifact, table_id)
        if fields.count("contrast_row_id") != 1 or fields.count("gain_display") != 1:
            raise SentimentClosenessReportError(
                f"packaged HTML view ID/display column drift: {table_id}"
            )
        id_index = fields.index("contrast_row_id")
        display_index = fields.index("gain_display")
        table = next(
            item
            for item in artifact["manifest"]["tables"]
            if item.get("id") == table_id
        )
        expected_rows = artifact["snapshot"]["datasets"].get(str(table["dataset"]))
        visible_rows = parser.rows.get(table_id, [])
        if (
            not isinstance(expected_rows, list)
            or len(visible_rows) != len(expected)
            or any(len(row) != len(fields) for row in visible_rows)
        ):
            raise SentimentClosenessReportError(
                f"packaged HTML view row coverage drift: {table_id}"
            )
        visible = [row[id_index] for row in visible_rows]
        expected_displays = {
            str(row["contrast_row_id"]): str(row["gain_display"])
            for row in expected_rows
        }
        visible_displays = {row[id_index]: row[display_index] for row in visible_rows}
        if (
            len(set(visible)) != len(visible)
            or set(visible) != set(expected)
            or visible_displays != expected_displays
        ):
            raise SentimentClosenessReportError(
                f"packaged HTML view identity/display drift: {table_id}"
            )
        all_visible.extend(visible)
    if (
        len(all_visible) != EXPECTED_VIEW_ROBUSTNESS_ROWS
        or len(set(all_visible)) != EXPECTED_VIEW_ROBUSTNESS_ROWS
    ):
        raise SentimentClosenessReportError(
            "packaged HTML view-robustness coverage drift"
        )
    return len(all_visible)


def _validate_visible_scatter(
    parser: _VisibleTableParser, artifact: Mapping[str, Any]
) -> dict[str, int]:
    _scatter_contract(artifact)
    datasets = artifact["snapshot"]["datasets"]
    counts = {"distil": 0, "finbert": 0}
    exact_cells = 0
    expected_chart_keys = set()
    for chart in artifact["manifest"]["charts"]:
        chart_id = str(chart.get("id", ""))
        backend = next(
            (
                prefix
                for prefix in counts
                if chart_id.startswith(f"{prefix}_raw_scatter_page_")
            ),
            None,
        )
        if backend is None:
            continue
        parser_key = f"chart:{chart_id}"
        expected_chart_keys.add(parser_key)
        fields = _chart_fields(chart)
        if fields.count("point_label") != 1 or any(
            fields.count(field) != 1 for field in SCATTER_EXACT_FIELDS.values()
        ):
            raise SentimentClosenessReportError(
                f"packaged HTML scatter exact column drift: {chart_id}"
            )
        point_index = fields.index("point_label")
        display_indexes = {
            display_field: fields.index(display_field)
            for display_field in SCATTER_EXACT_FIELDS.values()
        }
        expected_rows = datasets.get(str(chart.get("dataset")))
        visible_rows = parser.rows.get(parser_key, [])
        if (
            not isinstance(expected_rows, list)
            or len(visible_rows) != len(expected_rows)
            or any(len(row) != len(fields) for row in visible_rows)
        ):
            raise SentimentClosenessReportError(
                f"packaged HTML scatter row coverage drift: {chart_id}"
            )
        expected = {str(row["point_label"]): row for row in expected_rows}
        visible = {row[point_index]: row for row in visible_rows}
        if len(expected) != len(expected_rows) or set(visible) != set(expected):
            raise SentimentClosenessReportError(
                f"packaged HTML scatter point identity drift: {chart_id}"
            )
        for point_label, expected_row in expected.items():
            visible_row = visible[point_label]
            for display_field, index in display_indexes.items():
                if visible_row[index] != str(expected_row[display_field]):
                    raise SentimentClosenessReportError(
                        "packaged HTML lost exact scatter value: "
                        f"{chart_id}/{point_label}/{display_field}"
                    )
                exact_cells += 1
        counts[backend] += len(expected_rows)
    visible_chart_keys = {key for key in parser.rows if key.startswith("chart:")}
    if visible_chart_keys != expected_chart_keys or any(
        count != EXPECTED_SCATTER_ROWS_PER_BACKEND for count in counts.values()
    ):
        raise SentimentClosenessReportError(
            "packaged HTML scatter panel coverage drift"
        )
    return {**counts, "exact_cells": exact_cells}


def _looks_like_compacted_zero(value: str) -> bool:
    normalized = value.strip().replace("−", "-").replace(",", "")
    return bool(_COMPACT_ZERO_RE.fullmatch(normalized))


def _validate_visible_host_inventory(
    parser: _VisibleTableParser, artifact: Mapping[str, Any]
) -> dict[str, int]:
    inventory = {
        "tables": parser.table_hosts,
        "charts": parser.chart_hosts,
        "cards": parser.card_hosts,
    }
    result: dict[str, int] = {}
    for collection, observed in inventory.items():
        expected = {str(item["id"]): 1 for item in artifact["manifest"][collection]}
        if observed != expected:
            raise SentimentClosenessReportError(
                f"packaged HTML {collection} host inventory drift"
            )
        result[collection] = len(observed)
    return result


def validate_packaged_html(
    report_path: Path, artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed on portable fallback truncation, omission, or precision loss."""

    try:
        html = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SentimentClosenessReportError(
            f"cannot read packaged HTML: {exc}"
        ) from exc
    if "Showing first" in html:
        raise SentimentClosenessReportError(
            "packaged HTML contains a truncated semantic fallback"
        )
    parser = _VisibleTableParser()
    parser.feed(html)
    parser.close()
    visible_host_counts = _validate_visible_host_inventory(parser, artifact)

    expected_estimates_by_table = _appendix_ids_by_table(
        artifact,
        prefix="all_estimates",
        row_id="estimate_id",
        expected=EXPECTED_ESTIMATE_ROWS,
    )
    expected_contrasts_by_table = _appendix_ids_by_table(
        artifact,
        prefix="all_contrasts",
        row_id="contrast_row_id",
        expected=EXPECTED_CONTRAST_ROWS,
    )
    expected_estimates = {
        value for values in expected_estimates_by_table.values() for value in values
    }
    expected_contrasts = {
        value for values in expected_contrasts_by_table.values() for value in values
    }
    visible_estimates = _visible_ids(parser.rows, pattern=_ESTIMATE_TABLE_RE)
    visible_contrasts = _visible_ids(parser.rows, pattern=_CONTRAST_TABLE_RE)
    visible_estimate_tables = {
        table_id: [row[0] for row in rows if row]
        for table_id, rows in parser.rows.items()
        if _ESTIMATE_TABLE_RE.fullmatch(table_id)
    }
    visible_contrast_tables = {
        table_id: [row[0] for row in rows if row]
        for table_id, rows in parser.rows.items()
        if _CONTRAST_TABLE_RE.fullmatch(table_id)
    }
    estimate_pages_match = set(visible_estimate_tables) == set(
        expected_estimates_by_table
    ) and all(
        len(visible_estimate_tables[table_id]) == len(expected_ids)
        and set(visible_estimate_tables[table_id]) == set(expected_ids)
        for table_id, expected_ids in expected_estimates_by_table.items()
    )
    contrast_pages_match = set(visible_contrast_tables) == set(
        expected_contrasts_by_table
    ) and all(
        len(visible_contrast_tables[table_id]) == len(expected_ids)
        and set(visible_contrast_tables[table_id]) == set(expected_ids)
        for table_id, expected_ids in expected_contrasts_by_table.items()
    )
    if (
        len(visible_estimates) != EXPECTED_ESTIMATE_ROWS
        or len(set(visible_estimates)) != EXPECTED_ESTIMATE_ROWS
        or set(visible_estimates) != expected_estimates
        or not estimate_pages_match
    ):
        raise SentimentClosenessReportError(
            "packaged HTML estimate appendix is incomplete or duplicated"
        )
    if (
        len(visible_contrasts) != EXPECTED_CONTRAST_ROWS
        or len(set(visible_contrasts)) != EXPECTED_CONTRAST_ROWS
        or set(visible_contrasts) != expected_contrasts
        or not contrast_pages_match
    ):
        raise SentimentClosenessReportError(
            "packaged HTML contrast appendix is incomplete or duplicated"
        )
    expected_estimate_displays = _appendix_displays_by_table(
        artifact,
        prefix="all_estimates",
        row_id="estimate_id",
        numeric_field="estimate",
        display_field="estimate_display",
        expected=EXPECTED_ESTIMATE_ROWS,
    )
    expected_contrast_displays = _appendix_displays_by_table(
        artifact,
        prefix="all_contrasts",
        row_id="contrast_row_id",
        numeric_field="gain",
        display_field="gain_display",
        expected=EXPECTED_CONTRAST_ROWS,
    )
    visible_estimate_displays = _validate_visible_appendix_displays(
        parser=parser,
        artifact=artifact,
        expected_by_table=expected_estimate_displays,
        row_id="estimate_id",
        display_field="estimate_display",
        label="estimate",
    )
    visible_contrast_displays = _validate_visible_appendix_displays(
        parser=parser,
        artifact=artifact,
        expected_by_table=expected_contrast_displays,
        row_id="contrast_row_id",
        display_field="gain_display",
        label="gain",
    )
    expected_headline_points = _headline_point_contract(artifact)
    for card_id, expected_value in expected_headline_points.items():
        visible_values = parser.card_values.get(card_id, [])
        if not visible_values or visible_values[0] != expected_value:
            raise SentimentClosenessReportError(
                f"packaged HTML lost coercion-safe headline point: {card_id}"
            )
    visible_view_rows = _validate_visible_view_ids(parser, artifact)
    visible_scatter = _validate_visible_scatter(parser, artifact)

    expected_scales = artifact["snapshot"]["datasets"]["reference_scales"]
    visible_scales = parser.rows.get("reference_scales_table", [])
    if len(visible_scales) != len(expected_scales):
        raise SentimentClosenessReportError(
            "packaged HTML reference-scale row coverage drift"
        )
    visible_by_backend = {row[0]: row for row in visible_scales if row}
    for expected in expected_scales:
        visible = visible_by_backend.get(str(expected["backend"]))
        required = {
            str(expected["reference_mean_display"]),
            str(expected["reference_sd_display"]),
        }
        if visible is None or not required <= set(visible):
            raise SentimentClosenessReportError(
                f"packaged HTML lost exact reference scale: {expected['backend_id']}"
            )

    support_rows = artifact["snapshot"]["datasets"]["support_diagnostics"]
    expected_support = [
        row
        for row in support_rows
        if row.get("backend_id") == PRIMARY_BACKEND_ID
        and row.get("artifact_arm") == REFERENCE_ARM
    ]
    if len(expected_support) != 1:
        raise SentimentClosenessReportError(
            "Distil reference support artifact identity drift"
        )
    support_fields = _table_fields(artifact, "support_diagnostics_table")
    support_indexes = {field: index for index, field in enumerate(support_fields)}
    required_fields = {"backend", "artifact_arm", *SUPPORT_EXACT_FIELDS.values()}
    if not required_fields <= set(support_indexes):
        raise SentimentClosenessReportError(
            "Distil reference support table binding drift"
        )
    visible_support = parser.rows.get("support_diagnostics_table", [])
    if len(visible_support) != len(support_rows) or any(
        len(row) != len(support_fields) for row in visible_support
    ):
        raise SentimentClosenessReportError(
            "packaged HTML support-diagnostic row coverage drift"
        )
    expected_support_by_key = {
        (str(row["backend"]), str(row["artifact_arm"])): row for row in support_rows
    }
    visible_support_by_key = {
        (
            row[support_indexes["backend"]],
            row[support_indexes["artifact_arm"]],
        ): row
        for row in visible_support
    }
    if set(visible_support_by_key) != set(expected_support_by_key):
        raise SentimentClosenessReportError(
            "packaged HTML support-diagnostic row identity drift"
        )
    expected_distil_support = expected_support[0]
    distil_support_key = (
        str(expected_distil_support["backend"]),
        str(expected_distil_support["artifact_arm"]),
    )
    visible_distil_support_row = visible_support_by_key[distil_support_key]
    visible_support_mean = visible_distil_support_row[
        support_indexes["mean_raw_score_display"]
    ]
    visible_support_sd = visible_distil_support_row[
        support_indexes["raw_score_sd_display"]
    ]
    if _looks_like_compacted_zero(visible_support_mean) or _looks_like_compacted_zero(
        visible_support_sd
    ):
        raise SentimentClosenessReportError(
            "packaged HTML compacted Distil reference support mean/SD to -0 or 0"
        )
    expected_support_mean = str(expected_distil_support["mean_raw_score_display"])
    expected_support_sd = str(expected_distil_support["raw_score_sd_display"])
    support_exact_cells = 0
    for key, expected_row in expected_support_by_key.items():
        visible_row = visible_support_by_key[key]
        for display_field in SUPPORT_EXACT_FIELDS.values():
            if visible_row[support_indexes[display_field]] != str(
                expected_row[display_field]
            ):
                raise SentimentClosenessReportError(
                    "packaged HTML lost exact support diagnostic: "
                    f"{key[0]}/{key[1]}/{display_field}"
                )
            support_exact_cells += 1
    return {
        "html_sha256": _sha256_file(report_path),
        "showing_first_occurrences": 0,
        "visible_host_counts": visible_host_counts,
        "visible_estimate_rows": len(visible_estimates),
        "unique_visible_estimate_rows": len(set(visible_estimates)),
        "visible_estimate_pages": len(visible_estimate_tables),
        "visible_estimate_exact_displays": visible_estimate_displays,
        "sorted_estimate_ids_sha256": hashlib.sha256(
            _canonical(sorted(visible_estimates)).encode("utf-8")
        ).hexdigest(),
        "visible_contrast_rows": len(visible_contrasts),
        "unique_visible_contrast_rows": len(set(visible_contrasts)),
        "visible_contrast_pages": len(visible_contrast_tables),
        "visible_contrast_exact_displays": visible_contrast_displays,
        "sorted_contrast_ids_sha256": hashlib.sha256(
            _canonical(sorted(visible_contrasts)).encode("utf-8")
        ).hexdigest(),
        "visible_reference_scale_rows": len(visible_scales),
        "reference_scale_exact_text": True,
        "visible_headline_point_cards": len(expected_headline_points),
        "headline_point_exact_text": True,
        "visible_distil_scatter_rows": visible_scatter["distil"],
        "visible_finbert_scatter_rows": visible_scatter["finbert"],
        "visible_scatter_exact_cells": visible_scatter["exact_cells"],
        "visible_view_robustness_rows": visible_view_rows,
        "visible_support_rows": len(visible_support),
        "visible_support_exact_cells": support_exact_cells,
        "support_raw_exact_text": True,
        "distil_reference_support_mean_exact_text": expected_support_mean,
        "distil_reference_support_sd_exact_text": expected_support_sd,
    }


def render_and_seal_report(
    *,
    estimator_manifest: Path,
    estimator_manifest_sha256: str,
    output_dir: Path,
    node_bin: Path | None = None,
    plugin_root: Path | None = None,
) -> dict[str, Any]:
    """Deep-validate twice and publish exactly three immutable v2 files."""

    output = output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise SentimentClosenessReportError(
            f"report directory already exists: {output}"
        )
    renderer_binding = _binding(Path(__file__))
    base_renderer_binding = _binding(Path(_base.__file__))
    report_model, estimator_binding = load_report_model(
        estimator_manifest, estimator_manifest_sha256
    )
    artifact = build_canonical_artifact(report_model)
    artifact_contract = _validate_artifact_fallback_contract(artifact)
    artifact_text = _canonical(artifact)
    for required in (
        PAPER_STAGE_DISCLOSURE,
        REFERENCE_DISCLOSURE,
        MEETING_GRAIN_DISCLOSURE,
        IDENTITY_LINE_DISCLOSURE,
        NONPOOLED_DISCLOSURE,
    ):
        if required not in artifact_text:
            raise SentimentClosenessReportError(
                "canonical artifact lost a required disclosure"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    ).resolve()
    published = False
    try:
        artifact_path = staging / "artifact.json"
        report_path = staging / "report.html"
        _write_new_json(artifact_path, artifact)
        receipt, builder_bindings = _package_report(
            artifact_path=artifact_path,
            report_path=report_path,
            node_bin=node_bin,
            plugin_root=plugin_root,
        )
        html_acceptance = validate_packaged_html(report_path, artifact)

        refreshed_model, refreshed_binding = load_report_model(
            estimator_manifest, estimator_manifest_sha256
        )
        if refreshed_binding != estimator_binding or refreshed_model != report_model:
            raise SentimentClosenessReportError(
                "estimator bundle changed during reporting"
            )
        if build_canonical_artifact(refreshed_model) != artifact:
            raise SentimentClosenessReportError(
                "canonical v2 report artifact is not reproducible"
            )
        if (
            _binding(Path(__file__)) != renderer_binding
            or _binding(Path(_base.__file__)) != base_renderer_binding
        ):
            raise SentimentClosenessReportError(
                "v1 base or v2 renderer source changed during reporting"
            )

        artifact_final = output / "artifact.json"
        report_final = output / "report.html"
        manifest_final = output / "manifest.json"
        manifest = seal_manifest(
            {
                "schema_version": REPORT_MANIFEST_SCHEMA,
                "status": "complete",
                "immutable": True,
                "report_id": REPORT_ID,
                "created_at_utc": report_model["generated_at_utc"],
                "operation": "render_only_no_metric_or_bootstrap_recomputation",
                "estimator_manifest": copy.deepcopy(estimator_binding),
                "artifacts": {
                    "canonical_artifact": {
                        **_intended_binding(artifact_path, artifact_final),
                        "media_type": "application/json",
                        "contract": REPORT_ARTIFACT_CONTRACT,
                    },
                    "portable_report": {
                        **_intended_binding(report_path, report_final),
                        "media_type": "text/html; charset=utf-8",
                        "self_contained": True,
                        "network_dependencies": False,
                    },
                },
                "report_builder": builder_bindings,
                "delivery_receipt": receipt,
                "renderer": renderer_binding,
                "base_renderer": base_renderer_binding,
                "runtime": _report_runtime(),
                "report_contract": {
                    "audience": "technical",
                    "delivery_mode": "html",
                    "surface_count": 1,
                    "canonical_artifact_packaged_once": True,
                    "snapshot_status": "ready",
                    "bootstrap_draws": EXPECTED_BOOTSTRAP_DRAWS,
                    "primary_meeting_n": PRIMARY_N_MEETINGS,
                    "replicate_pooling": False,
                    "paper_chk2_artifact_chk3_alias_displayed": True,
                    "direct_metrics_and_chk1_gains_displayed": True,
                    "finbert_views_and_neutral_robustness_displayed": True,
                    "appendix_page_size": APPENDIX_PAGE_SIZE,
                    "fallback_fidelity": artifact_contract,
                    "html_acceptance": html_acceptance,
                    "chart_contract": {
                        "primary_scatter": (
                            "raw reference x versus shared-complete paper-CHK2 "
                            "replicate mean y; N=128 across three chronological "
                            "non-truncated panels"
                        ),
                        "robustness_scatter": (
                            "raw FinBERT reference x versus shared-complete "
                            "paper-CHK2 replicate mean y; N=128 across three "
                            "chronological non-truncated panels; non-pooled construct"
                        ),
                        "identity_line_omission_reason": (
                            "Canonical referenceLines are axis-constant and cannot "
                            "represent the diagonal y=x; adjacent text defines it."
                        ),
                    },
                },
                "limitations": [
                    REFERENCE_DISCLOSURE,
                    MEETING_GRAIN_DISCLOSURE,
                    NONPOOLED_DISCLOSURE,
                    (
                        "Browser-level chart QA was unavailable; semantic chart "
                        "tables remain present."
                        if receipt["stages"]["verification"] == "structural_only"
                        else "Browser-level desktop and narrow-width QA passed."
                    ),
                ],
            }
        )
        _write_new_json(staging / "manifest.json", manifest)
        validate_manifest_integrity(manifest)
        try:
            _base._delivery._seal_permissions(staging)
            _base._delivery._fsync_directory(staging)
            _base._delivery._rename_noreplace(staging, output)
        except _base._delivery.BetaTechnicalReportError as exc:
            raise SentimentClosenessReportError(str(exc)) from exc
        published = True
        _base._delivery._fsync_directory(output.parent)

        final_manifest = json.loads(manifest_final.read_text(encoding="utf-8"))
        validate_manifest_integrity(final_manifest)
        if {path.name for path in output.iterdir()} != {
            "artifact.json",
            "report.html",
            "manifest.json",
        }:
            raise SentimentClosenessReportError("published report inventory drift")
        for name, bound in final_manifest["artifacts"].items():
            observed = _binding(Path(str(bound["path"])))
            if any(
                observed[field] != bound[field] for field in ("path", "bytes", "sha256")
            ):
                raise SentimentClosenessReportError(f"published {name} binding drift")
        if output.stat().st_mode & 0o222:
            raise SentimentClosenessReportError("published report root is writable")
        if validate_packaged_html(report_final, artifact) != html_acceptance:
            raise SentimentClosenessReportError("published HTML acceptance drift")
        return final_manifest
    finally:
        if not published and staging.exists():
            for path in [*staging.rglob("*"), staging]:
                try:
                    path.chmod(0o755 if path.is_dir() else 0o600)
                except OSError:
                    pass
            shutil.rmtree(staging, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    describe = subparsers.add_parser(
        "describe-contract", help="print the result-free v2 adapter contract"
    )
    describe.set_defaults(command="describe-contract")
    render = subparsers.add_parser(
        "render", help="deep-validate the estimator twice and publish v2"
    )
    render.add_argument(
        "--estimator-manifest", type=Path, default=DEFAULT_ESTIMATOR_MANIFEST
    )
    render.add_argument("--estimator-manifest-sha256", required=True)
    render.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    render.add_argument("--node-bin", type=Path)
    render.add_argument("--plugin-root", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "describe-contract":
        print(json.dumps(_normalized_contract(), ensure_ascii=False, indent=2))
        return 0
    try:
        manifest = render_and_seal_report(
            estimator_manifest=args.estimator_manifest,
            estimator_manifest_sha256=args.estimator_manifest_sha256,
            output_dir=args.output_dir,
            node_bin=args.node_bin,
            plugin_root=args.plugin_root,
        )
    except (
        SentimentClosenessReportError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ERROR: {exc}", file=_base.sys.stderr)
        return 1
    print(
        _canonical(
            {
                "status": manifest["status"],
                "report": manifest["artifacts"]["portable_report"]["path"],
                "report_sha256": manifest["artifacts"]["portable_report"]["sha256"],
                "manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
                "verification": manifest["delivery_receipt"]["stages"]["verification"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
