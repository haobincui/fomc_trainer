"""Render a sealed, self-contained technical report for CHK3 bootstrap scores.

The renderer performs no generation, semantic inference, or statistical
recomputation.  It deep-validates the frozen score bundle, displays the exact
point estimates/CIs/paired tests already sealed by the scorer, retains the
deterministic greedy N12 anchor as a separate descriptive table, and publishes
one portable HTML file plus a sealed report manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


REPORT_MANIFEST_SCHEMA_VERSION = "chk3-stochastic-bootstrap-portable-report-manifest-v1"
SCORE_MANIFEST_SCHEMA_VERSION = "chk3-stochastic-bootstrap-scorecard-v1"
RESULT_SCHEMA_VERSION = "chk3-stochastic-bootstrap-results-v1"
ROW_SCHEMA_VERSION = "chk3-stochastic-bootstrap-six-metric-row-v1"
DRAW_SCHEMA_VERSION = "chk3-stochastic-bootstrap-draw-v1"
FAILURE_SCHEMA_VERSION = "chk3-stochastic-bootstrap-failure-v1"
GREEDY_SCHEMA_VERSION = "chk0-chk1-chk3-six-metrics-v1"

MODEL_ORDER = ("chk0", "chk1", "chk3")
VIEW_ORDER = ("full", "row_disjoint", "strict_meeting_disjoint")
VIEW_LABELS = {
    "full": "Full held-out test（描述性）",
    "row_disjoint": "Row-disjoint（主稳健性视图）",
    "strict_meeting_disjoint": "Strict meeting-disjoint（敏感性）",
}
METRIC_ORDER = (
    "structure_delivery",
    "numeric_fidelity",
    "date_fidelity",
    "degeneration_free",
    "mpnet_cosine",
    "bertscore_f1",
)
METRIC_LABELS = {
    "structure_delivery": "结构 / 交付",
    "numeric_fidelity": "数值保真",
    "date_fidelity": "日期保真",
    "degeneration_free": "无退化",
    "mpnet_cosine": "MPNet cosine",
    "bertscore_f1": "BERTScore-F1",
}
EXPECTED_ROW_SCORES = 2_850
EXPECTED_DRAW_ROWS = 6_000
EXPECTED_ROWS_PER_MODEL = 950
EXPECTED_DRAWS_PER_VIEW = 2_000
EXPECTED_GREEDY_PAYLOAD_SHA256 = (
    "2828a7c6d94ec586b66c7edae12f9f31674cc528bcd346198c448d45b1322e72"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VIEW_DENOMINATORS = {
    "full": (190, 171, 13, False),
    "row_disjoint": (178, 160, 13, True),
    "strict_meeting_disjoint": (59, 52, 4, False),
}


class BootstrapReportError(RuntimeError):
    """A sealed input or portable-report invariant failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise BootstrapReportError(f"non-canonical JSON payload: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise BootstrapReportError(f"missing regular JSON file: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapReportError(f"cannot read JSON {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BootstrapReportError(f"JSON root is not an object: {resolved}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise BootstrapReportError(f"missing regular JSONL file: {resolved}")
    rows: list[dict[str, Any]] = []
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise BootstrapReportError(
                        f"blank JSONL row: {resolved}:{line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise BootstrapReportError(
                        f"JSONL row is not an object: {resolved}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapReportError(f"cannot read JSONL {resolved}: {exc}") from exc
    return rows


def _file_binding(path: Path, *, sealed: bool = False) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise BootstrapReportError(f"artifact is not a regular file: {resolved}")
    binding: dict[str, Any] = {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if sealed:
        try:
            binding["payload_sha256"] = validate_manifest_integrity(
                _read_json(resolved)
            )
        except Exception as exc:
            raise BootstrapReportError(
                f"sealed artifact integrity failed for {resolved}: {exc}"
            ) from exc
    return binding


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_text(path: Path, text: str) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        with resolved.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise BootstrapReportError(f"refusing to overwrite report: {resolved}") from exc
    _fsync_directory(resolved.parent)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _binding_equal(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    keys = ("path", "sha256", "bytes", "payload_sha256")
    return all(observed.get(key) == expected.get(key) for key in keys if key in expected)


def _validate_artifact_binding(
    binding: Mapping[str, Any], *, name: str, parent: Path, sealed: bool = False
) -> Path:
    path = Path(str(binding.get("path"))).expanduser().resolve()
    if path.parent != parent:
        raise BootstrapReportError(f"{name} is outside the score directory")
    observed = _file_binding(path, sealed=sealed)
    if not _binding_equal(observed, binding):
        raise BootstrapReportError(f"{name} file binding drift")
    return path


def _validate_row_scores(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != EXPECTED_ROW_SCORES:
        raise BootstrapReportError(
            f"row score matrix must contain {EXPECTED_ROW_SCORES} rows"
        )
    keys: set[tuple[str, str, int]] = set()
    model_counts: Counter[str] = Counter()
    for row in rows:
        model_id = row.get("model_id")
        sample_id = row.get("sample_id")
        replicate_id = row.get("replicate_id")
        metrics = row.get("six_metrics")
        if (
            row.get("schema_version") != ROW_SCHEMA_VERSION
            or model_id not in MODEL_ORDER
            or not isinstance(sample_id, str)
            or isinstance(replicate_id, bool)
            or not isinstance(replicate_id, int)
            or not 0 <= replicate_id < 5
            or not isinstance(metrics, Mapping)
            or set(metrics) != set(METRIC_ORDER)
        ):
            raise BootstrapReportError("row score schema/tuple/metric drift")
        key = (str(model_id), sample_id, replicate_id)
        if key in keys:
            raise BootstrapReportError(f"duplicate score tuple: {key}")
        keys.add(key)
        model_counts[str(model_id)] += 1
    if model_counts != Counter({model_id: EXPECTED_ROWS_PER_MODEL for model_id in MODEL_ORDER}):
        raise BootstrapReportError(f"row score model coverage drift: {model_counts}")


def _validate_failures(
    failures: Sequence[Mapping[str, Any]], *, score_manifest: Mapping[str, Any]
) -> None:
    expected = score_manifest.get("coverage", {}).get("hard_gate_failure_rows")
    if len(failures) != expected:
        raise BootstrapReportError("failure JSONL count differs from score manifest")
    seen: set[tuple[str, str, int]] = set()
    for row in failures:
        key_record = row.get("tuple_key")
        if (
            row.get("schema_version") != FAILURE_SCHEMA_VERSION
            or not isinstance(key_record, Mapping)
        ):
            raise BootstrapReportError("failure row schema drift")
        key = (
            str(key_record.get("model_id")),
            str(key_record.get("sample_id")),
            int(key_record.get("replicate_id")),
        )
        if key in seen:
            raise BootstrapReportError(f"duplicate failure tuple: {key}")
        seen.add(key)


def _validate_draws(draws: Sequence[Mapping[str, Any]]) -> None:
    if len(draws) != EXPECTED_DRAW_ROWS:
        raise BootstrapReportError(
            f"bootstrap draw artifact must contain {EXPECTED_DRAW_ROWS} rows"
        )
    counts: Counter[str] = Counter()
    keys: set[tuple[str, int]] = set()
    for row in draws:
        view_id = row.get("view_id")
        draw_id = row.get("draw_id")
        if (
            row.get("schema_version") != DRAW_SCHEMA_VERSION
            or view_id not in VIEW_ORDER
            or isinstance(draw_id, bool)
            or not isinstance(draw_id, int)
            or not 0 <= draw_id < EXPECTED_DRAWS_PER_VIEW
            or not isinstance(row.get("model_metrics"), Mapping)
        ):
            raise BootstrapReportError("bootstrap draw schema/key drift")
        key = (str(view_id), draw_id)
        if key in keys:
            raise BootstrapReportError(f"duplicate bootstrap draw: {key}")
        keys.add(key)
        counts[str(view_id)] += 1
    expected = Counter(
        {view_id: EXPECTED_DRAWS_PER_VIEW for view_id in VIEW_ORDER}
    )
    if counts != expected:
        raise BootstrapReportError(f"bootstrap draw view coverage drift: {counts}")


def _validate_result_views(views: Mapping[str, Any]) -> None:
    if set(views) != set(VIEW_ORDER):
        raise BootstrapReportError("bootstrap result view inventory drift")
    for view_id in VIEW_ORDER:
        value = views[view_id]
        if not isinstance(value, Mapping):
            raise BootstrapReportError(f"{view_id} result is not an object")
        cohort = value.get("view")
        estimates = value.get("estimates")
        contrasts = value.get("contrasts")
        interpretation = value.get("interpretation")
        plan = value.get("bootstrap_index_plan")
        expected_generation, expected_semantic, expected_meetings, authorized = (
            VIEW_DENOMINATORS[view_id]
        )
        if (
            not isinstance(cohort, Mapping)
            or cohort.get("view_id") != view_id
            or cohort.get("generation_prompts") != expected_generation
            or cohort.get("semantic_prompts") != expected_semantic
            or cohort.get("meetings") != expected_meetings
            or cohort.get("inferential_conclusion_authorized") is not authorized
            or not isinstance(estimates, Mapping)
            or set(estimates) != set(MODEL_ORDER)
            or not isinstance(contrasts, list)
            or len(contrasts) != 18
            or not isinstance(interpretation, Mapping)
            or interpretation.get("verdict")
            not in {
                "random_decoding_regression",
                "semantic_increment_under_random_decoding",
                "robust_but_increment_uncertain",
            }
            or not isinstance(plan, Mapping)
            or SHA256_RE.fullmatch(str(plan.get("sha256"))) is None
        ):
            raise BootstrapReportError(f"{view_id} cohort/result contract drift")
        for model_id in MODEL_ORDER:
            model_estimates = estimates[model_id]
            if not isinstance(model_estimates, Mapping) or set(model_estimates) != set(
                METRIC_ORDER
            ):
                raise BootstrapReportError(f"{view_id}/{model_id} estimate inventory drift")
            for metric in METRIC_ORDER:
                record = model_estimates[metric]
                if not isinstance(record, Mapping):
                    raise BootstrapReportError("estimate record is not an object")
                for field in ("point_estimate", "ci_lower", "ci_upper"):
                    number = record.get(field)
                    if isinstance(number, bool) or not isinstance(number, (int, float)):
                        raise BootstrapReportError("estimate value is not numeric")
                    if not math.isfinite(float(number)):
                        raise BootstrapReportError("estimate value is not finite")
        family_counts = Counter(str(record.get("family")) for record in contrasts)
        if family_counts != Counter({"primary": 6, "background": 12}):
            raise BootstrapReportError(f"{view_id} contrast family size drift")
        primary_metrics = {
            str(record.get("metric"))
            for record in contrasts
            if record.get("contrast_id") == "chk3_minus_chk1"
        }
        if primary_metrics != set(METRIC_ORDER):
            raise BootstrapReportError(f"{view_id} primary contrast metric drift")
        expected_combinations = 16 if view_id == "strict_meeting_disjoint" else 8192
        if any(
            record.get("sign_flip_combinations") != expected_combinations
            for record in contrasts
        ):
            raise BootstrapReportError(f"{view_id} exact sign-flip size drift")


def load_and_validate_bundle(
    score_manifest_path: Path, expected_score_sha256: str
) -> dict[str, Any]:
    """Deep-validate the score manifest and every report input artifact."""

    score_path = score_manifest_path.expanduser().resolve()
    score_binding = _file_binding(score_path, sealed=True)
    if score_binding["sha256"] != expected_score_sha256:
        raise BootstrapReportError("score manifest external SHA-256 drift")
    score = _read_json(score_path)
    if (
        score.get("schema_version") != SCORE_MANIFEST_SCHEMA_VERSION
        or score.get("status") != "complete"
        or score.get("model_order") != list(MODEL_ORDER)
        or score.get("metric_order") != list(METRIC_ORDER)
        or score.get("coverage", {}).get("generation_rows") != EXPECTED_ROW_SCORES
        or score.get("coverage", {}).get("paired_tuple_coverage_identical") is not True
        or score.get("coverage", {}).get("input_truncation_rows") != 0
        or score.get("scoring_contract", {}).get("weighted_composite_calculated")
        is not False
    ):
        raise BootstrapReportError("score manifest identity/coverage contract drift")
    artifacts = score.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "row_scores",
        "failure_samples",
        "bootstrap_draws",
        "bootstrap_results",
    }:
        raise BootstrapReportError("score artifact inventory drift")
    score_dir = score_path.parent
    paths = {
        name: _validate_artifact_binding(
            artifacts[name],
            name=name,
            parent=score_dir,
            sealed=name == "bootstrap_results",
        )
        for name in artifacts
    }
    if (
        artifacts["row_scores"].get("rows") != EXPECTED_ROW_SCORES
        or artifacts["bootstrap_draws"].get("rows") != EXPECTED_DRAW_ROWS
        or artifacts["failure_samples"].get("rows")
        != score.get("coverage", {}).get("hard_gate_failure_rows")
    ):
        raise BootstrapReportError("score artifact declared row counts drift")
    row_scores = _read_jsonl(paths["row_scores"])
    failures = _read_jsonl(paths["failure_samples"])
    draws = _read_jsonl(paths["bootstrap_draws"])
    results = _read_json(paths["bootstrap_results"])
    _validate_row_scores(row_scores)
    _validate_failures(failures, score_manifest=score)
    _validate_draws(draws)
    try:
        results_payload_sha = validate_manifest_integrity(results)
    except Exception as exc:
        raise BootstrapReportError(f"bootstrap result integrity failed: {exc}") from exc
    if (
        results.get("schema_version") != RESULT_SCHEMA_VERSION
        or results.get("status") != "complete"
        or results.get("model_order") != list(MODEL_ORDER)
        or results.get("metric_order") != list(METRIC_ORDER)
        or results.get("view_order") != list(VIEW_ORDER)
        or score.get("scoring_contract", {}).get(
            "bootstrap_results_payload_sha256"
        )
        != results_payload_sha
        or score.get("scoring_contract", {}).get("bootstrap_draws_sha256")
        != artifacts["bootstrap_draws"].get("sha256")
    ):
        raise BootstrapReportError("bootstrap result/score binding drift")
    result_views = results.get("views")
    if not isinstance(result_views, Mapping):
        raise BootstrapReportError("bootstrap result views are missing")
    _validate_result_views(result_views)

    greedy_binding = score.get("inputs", {}).get("greedy_n12_six_metric_anchor")
    if not isinstance(greedy_binding, Mapping):
        raise BootstrapReportError("score manifest lost greedy anchor binding")
    greedy_path = Path(str(greedy_binding.get("path"))).expanduser().resolve()
    observed_greedy = _file_binding(greedy_path, sealed=True)
    if not _binding_equal(observed_greedy, greedy_binding):
        raise BootstrapReportError("greedy N12 anchor binding drift")
    greedy = _read_json(greedy_path)
    if (
        greedy.get("schema_version") != GREEDY_SCHEMA_VERSION
        or greedy.get("status") != "complete"
        or observed_greedy.get("payload_sha256")
        != EXPECTED_GREEDY_PAYLOAD_SHA256
        or len(greedy.get("row_scores", [])) != 36
    ):
        raise BootstrapReportError("greedy N12 anchor contract drift")
    return {
        "score": score,
        "score_binding": score_binding,
        "results": results,
        "row_scores": row_scores,
        "failures": failures,
        "draws": draws,
        "greedy": greedy,
        "greedy_binding": observed_greedy,
        "artifact_paths": paths,
    }


def _number(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "—"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "—"
    if numeric == 0:
        return "0"
    if abs(numeric) < 0.0001:
        return f"{numeric:.2e}"
    return f"{numeric:.{digits}f}"


def _ci(record: Mapping[str, Any]) -> str:
    return f"[{_number(record.get('ci_lower'))}, {_number(record.get('ci_upper'))}]"


def _verdict_label(value: str) -> str:
    labels = {
        "random_decoding_regression": "随机解码退化",
        "semantic_increment_under_random_decoding": "硬门禁不退化，语义增量获得支持",
        "robust_but_increment_uncertain": "稳健但增量不确定",
    }
    return labels.get(value, value)


def _forest_svg(view: Mapping[str, Any]) -> str:
    primary = {
        record["metric"]: record
        for record in view["contrasts"]
        if record["contrast_id"] == "chk3_minus_chk1"
    }
    records = [primary[metric] for metric in METRIC_ORDER]
    observed = [
        float(record[key])
        for record in records
        for key in ("ci_lower", "ci_upper", "point_difference")
    ]
    lower = min(-0.02, min(observed))
    upper = max(0.02, max(observed))
    padding = max(0.01, (upper - lower) * 0.08)
    lower -= padding
    upper += padding
    width, left, right, row_height = 900, 205, 32, 42
    plot_width = width - left - right
    height = 54 + row_height * len(records)

    def x(value: float) -> float:
        return left + (value - lower) / (upper - lower) * plot_width

    parts = [
        f'<svg class="forest" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="CHK3 minus CHK1 paired bootstrap confidence intervals">',
        f'<line class="zero" x1="{x(0):.2f}" x2="{x(0):.2f}" y1="22" y2="{height - 25}"/>',
    ]
    for index, (metric, record) in enumerate(zip(METRIC_ORDER, records, strict=True)):
        y = 39 + index * row_height
        ci_lower = float(record["ci_lower"])
        ci_upper = float(record["ci_upper"])
        point = float(record["point_difference"])
        parts.extend(
            [
                f'<text class="metric-label" x="4" y="{y + 5}">{html.escape(METRIC_LABELS[metric])}</text>',
                f'<line class="ci" x1="{x(ci_lower):.2f}" x2="{x(ci_upper):.2f}" y1="{y}" y2="{y}"/>',
                f'<circle class="point" cx="{x(point):.2f}" cy="{y}" r="5"/>',
                f'<text class="value-label" x="{width - right + 4}" y="{y + 5}">{html.escape(_number(point))}</text>',
            ]
        )
    parts.extend(
        [
            f'<text class="axis-label" x="{left}" y="{height - 6}">{html.escape(_number(lower, digits=3))}</text>',
            f'<text class="axis-label" text-anchor="end" x="{width - right}" y="{height - 6}">{html.escape(_number(upper, digits=3))}</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _estimate_table(view: Mapping[str, Any]) -> str:
    header = "".join(f"<th>{model.upper()}</th>" for model in MODEL_ORDER)
    rows: list[str] = []
    for metric in METRIC_ORDER:
        cells = []
        for model in MODEL_ORDER:
            record = view["estimates"][model][metric]
            failure_count = record.get("zero_or_fail_generation_count")
            cells.append(
                "<td>"
                f"<strong>{html.escape(_number(record['point_estimate']))}</strong> "
                f"<span class=\"muted\">{html.escape(_ci(record))}</span>"
                f"<br><small>失败生成 {html.escape(str(failure_count))} / {record['observations']}</small>"
                "</td>"
            )
        rows.append(
            f"<tr><th>{html.escape(METRIC_LABELS[metric])}</th>{''.join(cells)}</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>指标</th>'
        f"{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _contrast_table(view: Mapping[str, Any], *, family: str) -> str:
    records = [record for record in view["contrasts"] if record["family"] == family]
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['contrast_id'])}</td>"
            f"<td>{html.escape(METRIC_LABELS[record['metric']])}</td>"
            f"<td>{html.escape(_number(record['point_difference']))}</td>"
            f"<td>{html.escape(_ci(record))}</td>"
            f"<td>{html.escape(_number(record['p_value']))}</td>"
            f"<td>{html.escape(_number(record['holm_adjusted_p']))}</td>"
            f"<td>{record['sign_flip_combinations']}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>配对比较</th><th>指标</th><th>差值</th><th>95% CI</th>"
        "<th>exact p</th><th>Holm p</th><th>符号组合</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _greedy_table(greedy: Mapping[str, Any]) -> str:
    rows = []
    for metric in METRIC_ORDER:
        cells = []
        for model in MODEL_ORDER:
            record = greedy["summaries"][model]["cohorts"]["all_n12"][
                "six_metrics"
            ][metric]
            cells.append(f"<td>{html.escape(_number(record['score']))}</td>")
        rows.append(
            f"<tr><th>{html.escape(METRIC_LABELS[metric])}</th>{''.join(cells)}</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Greedy N12 指标</th>'
        + "".join(f"<th>{model.upper()}</th>" for model in MODEL_ORDER)
        + f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _failure_summary(failures: Sequence[Mapping[str, Any]]) -> str:
    by_model: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    by_metric: Counter[str] = Counter()
    for row in failures:
        by_model[str(row["model_id"])] += 1
        by_reason.update(str(value) for value in row["preregistered_core_failures"])
        by_metric.update(str(value) for value in row["failed_generation_metrics"])
    rows = "".join(
        f"<tr><td>{html.escape(model.upper())}</td><td>{by_model[model]}</td></tr>"
        for model in MODEL_ORDER
    )
    reasons = ", ".join(
        f"{html.escape(reason)}={count}" for reason, count in sorted(by_reason.items())
    ) or "无"
    metrics = ", ".join(
        f"{html.escape(METRIC_LABELS.get(metric, metric))}={count}"
        for metric, count in sorted(by_metric.items())
    ) or "无"
    examples = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['model_id']).upper())}</td>"
        f"<td>{html.escape(str(row['sample_id']))}</td>"
        f"<td>{row['replicate_id']}</td>"
        f"<td>{html.escape(', '.join(row['preregistered_core_failures']))}</td>"
        "</tr>"
        for row in failures[:25]
    )
    example_table = (
        '<div class="table-wrap"><table><thead><tr><th>模型</th><th>sample_id</th>'
        f"<th>replicate</th><th>原因</th></tr></thead><tbody>{examples}</tbody></table></div>"
        if examples
        else '<p class="ok">没有 hard-gate 失败 tuple。</p>'
    )
    return (
        '<div class="grid two"><div><h3>按模型</h3><table><thead><tr><th>模型</th>'
        f"<th>失败生成</th></tr></thead><tbody>{rows}</tbody></table></div>"
        f"<div><h3>原因汇总</h3><p><strong>指标：</strong>{metrics}</p>"
        f"<p><strong>规则：</strong>{reasons}</p></div></div>"
        '<h3>失败 tuple（前 25 条）</h3>'
        f"{example_table}"
    )


def render_report_html(
    *,
    score: Mapping[str, Any],
    results: Mapping[str, Any],
    greedy: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Mapping[str, Any]],
    title: str,
) -> str:
    """Return a self-contained UTF-8 HTML technical report."""

    primary_view = results["views"]["row_disjoint"]
    interpretation = primary_view["interpretation"]
    verdict = _verdict_label(str(interpretation["verdict"]))
    hard_regressions = interpretation.get("hard_gate_regressions", [])
    regression_text = (
        "、".join(METRIC_LABELS[item] for item in hard_regressions)
        if hard_regressions
        else "无"
    )
    view_sections = []
    for view_id in VIEW_ORDER:
        view = results["views"][view_id]
        cohort = view["view"]
        view_verdict = _verdict_label(str(view["interpretation"]["verdict"]))
        inference_note = (
            "该视图是预先指定的主稳健性结论视图。"
            if cohort["inferential_conclusion_authorized"]
            else (
                "该视图仅作描述，不替代 row-disjoint 主结论。"
                if view_id == "full"
                else "仅作敏感性检查；4 个 meeting 不作显著性结论。"
            )
        )
        view_sections.append(
            f"""
            <section id="{html.escape(view_id)}">
              <div class="section-kicker">{html.escape(VIEW_LABELS[view_id])}</div>
              <h2>{html.escape(view_verdict)}</h2>
              <p class="lead">generation N={cohort['generation_prompts']}；semantic N={cohort['semantic_prompts']}；meeting={cohort['meetings']}。{html.escape(inference_note)}</p>
              {_forest_svg(view)}
              <p class="evidence-note">图：CHK3 cp318 − CHK1 的 paired bootstrap 差值与 95% percentile CI；竖线为 0。所有模型与六项指标共享 meeting/replicate 抽样索引。</p>
              <p class="evidence-note"><strong>Bootstrap index plan SHA-256：</strong><code>{html.escape(str(view['bootstrap_index_plan']['sha256']))}</code></p>
              <h3>模型点估计与区间</h3>
              {_estimate_table(view)}
              <h3>主比较：CHK3 − CHK1（Holm family=6）</h3>
              {_contrast_table(view, family='primary')}
              <details><summary>背景比较：CHK1 − CHK0、CHK3 − CHK0（Holm family=12）</summary>{_contrast_table(view, family='background')}</details>
            </section>
            """
        )

    source_rows = "".join(
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f"<td><code>{html.escape(str(binding.get('sha256', '—')))}</code></td>"
        f"<td><code>{html.escape(str(binding.get('payload_sha256', '—')))}</code></td>"
        "</tr>"
        for label, binding in bindings.items()
    )
    limitations = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in results["limitations"]
    )
    bootstrap = results["bootstrap_contract"]
    generated = _utc_now()
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="generator" content="render_chk3_stochastic_bootstrap_report.py">
  <title>{html.escape(title)}</title>
  <style>
    :root{{--ink:#17202a;--muted:#5d6d7e;--line:#d7dde5;--paper:#fff;--wash:#f4f7fa;--accent:#175c8f;--good:#176b4d;--warn:#986a00;--bad:#a52a2a}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}
    main{{max-width:1180px;margin:auto;background:var(--paper);padding:48px 54px 72px;box-shadow:0 0 28px #22334418}}
    h1{{font-size:36px;line-height:1.15;margin:.2em 0}} h2{{font-size:25px;margin:.25em 0 .45em}} h3{{font-size:18px;margin:1.4em 0 .5em}}
    section{{border-top:1px solid var(--line);padding-top:30px;margin-top:36px}} .eyebrow,.section-kicker{{color:var(--accent);font-weight:700;letter-spacing:.05em;text-transform:uppercase;font-size:12px}}
    .lead{{font-size:17px;color:#34495e}} .verdict{{border-left:6px solid var(--accent);background:#eef5fa;padding:20px 24px;margin:24px 0}}
    .verdict strong{{font-size:22px}} .grid{{display:grid;gap:18px}} .grid.two{{grid-template-columns:repeat(2,minmax(0,1fr))}}
    .card{{border:1px solid var(--line);border-radius:8px;padding:18px;background:#fff}} .card .value{{font-size:27px;font-weight:700}}
    table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} th,td{{border-bottom:1px solid var(--line);padding:9px 10px;text-align:right;vertical-align:top}} th:first-child,td:first-child{{text-align:left}} thead th{{background:#edf2f6;position:sticky;top:0}}
    .table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:6px}} .muted,.evidence-note,small{{color:var(--muted)}} .evidence-note{{font-size:13px}}
    details{{margin-top:18px}} summary{{cursor:pointer;color:var(--accent);font-weight:700}} code{{font-size:12px;overflow-wrap:anywhere}} .ok{{color:var(--good);font-weight:700}}
    .forest{{width:100%;height:auto;border:1px solid var(--line);background:#fbfcfd;border-radius:6px}} .forest .zero{{stroke:#8c98a4;stroke-width:1.5;stroke-dasharray:5 4}} .forest .ci{{stroke:var(--accent);stroke-width:3}} .forest .point{{fill:var(--accent)}} .forest text{{font:13px system-ui,sans-serif;fill:var(--ink)}} .forest .value-label{{font-variant-numeric:tabular-nums}} .forest .axis-label{{fill:var(--muted);font-size:11px}}
    footer{{margin-top:48px;border-top:1px solid var(--line);padding-top:20px;color:var(--muted);font-size:12px}}
    @media(max-width:760px){{main{{padding:28px 18px}}.grid.two{{grid-template-columns:1fr}}h1{{font-size:29px}}}}
    @media print{{body{{background:#fff}}main{{box-shadow:none;max-width:none;padding:20px}}details{{display:block}}}}
  </style>
</head>
<body><main>
  <header>
    <div class="eyebrow">CHK3 post-selection robustness · N190 × K5 · GPU0 generation</div>
    <h1>{html.escape(title)}</h1>
    <p class="lead">技术报告 · 生成于 {html.escape(generated)} · 统计结果来自已封存 artifacts，报告不重算指标或 bootstrap。</p>
  </header>

  <section>
    <div class="section-kicker">技术摘要</div>
    <h2>主结论：{html.escape(verdict)}</h2>
    <div class="verdict"><strong>{html.escape(verdict)}</strong><br>Row-disjoint 主视图的 hard-gate 下降项：{html.escape(regression_text)}。语义增量支持={str(bool(interpretation['semantic_increment_supported'])).lower()}。</div>
    <div class="grid two">
      <div class="card"><div class="muted">随机生成矩阵</div><div class="value">2,850</div><div>190 prompts × 3 models × 5 replicates</div></div>
      <div class="card"><div class="muted">层级 paired bootstrap</div><div class="value">{bootstrap['draws']:,}</div><div>seed={bootstrap['seed']}；meeting 等权</div></div>
      <div class="card"><div class="muted">Row-disjoint 分母</div><div class="value">178 / 160</div><div>generation / semantic；13 meetings</div></div>
      <div class="card"><div class="muted">Hard-gate 退化规则</div><div class="value">不可抵消</div><div>MPNet/BERT 提升不能抵消任一硬门禁下降</div></div>
    </div>
  </section>

  {''.join(view_sections)}

  <section id="greedy-anchor">
    <div class="section-kicker">确定性 anchor</div>
    <h2>旧 greedy N12 仅作为描述性锚点</h2>
    <p class="lead">该表来自冻结 cp318 六项 scorecard；它未进入随机 bootstrap，也不能扩大随机实验的有效样本量。</p>
    {_greedy_table(greedy)}
  </section>

  <section id="failures">
    <div class="section-kicker">失败审计</div>
    <h2>Hard-gate 失败按 tuple 原样保留</h2>
    {_failure_summary(failures)}
    <p class="evidence-note">完整失败清单保存在 sealed score bundle 的 failure_samples.jsonl；本页仅显示前 25 条 tuple。</p>
  </section>

  <section id="definitions">
    <div class="section-kicker">范围与定义</div>
    <h2>六项指标、三个 cohort、两个推断 family 均预先固定</h2>
    <ul>
      <li>Hard gates：结构/交付、数值保真、日期保真、无退化；任一失败时非 identity 行的两项语义指标固定为 0。</li>
      <li>Semantic：MPNet cosine 与 BERTScore-F1；19 条 normalized-identity prompt 不进入语义分母。</li>
      <li>Full：generation N190 / semantic N171；Row-disjoint：N178 / N160；Strict meeting-disjoint：N59 / N52、4 meetings。</li>
      <li>主 family 为 CHK3−CHK1 六项 Holm；背景 family 为 CHK1−CHK0 与 CHK3−CHK0 共十二项 Holm。</li>
    </ul>
  </section>

  <section id="method">
    <div class="section-kicker">实验与统计设计</div>
    <h2>共享索引维持跨模型配对，meeting 是主抽样单位</h2>
    <ol>
      <li>每条 prompt 先平均 5 个随机生成；meeting 内平均 prompt；meeting 之间等权。</li>
      <li>每个 bootstrap draw 有放回抽取 M 个 meetings；保留每个 meeting 的全部 rows，不做 row resampling。</li>
      <li>每个 row occurrence 有放回抽取 5 个 replicate；三模型和六指标共享同一 meeting/replicate 索引。</li>
      <li>报告 95% percentile CI；13 meetings 使用 exact 8,192 sign flips，strict 4 meetings 使用 16 种组合但不作显著性结论。</li>
    </ol>
  </section>

  <section id="limitations">
    <div class="section-kicker">不确定性与稳健性边界</div>
    <h2>Bootstrap 不能消除 cp318 的选点偏差</h2>
    <ul>{limitations}</ul>
    <p><strong>结论边界：</strong>这是 post-selection robustness，不是新增 meeting 的独立确认；semantic similarity 也不是事实正确性的替代指标。</p>
  </section>

  <section id="next-steps">
    <div class="section-kicker">建议的下一步</div>
    <h2>用新增 meeting 做独立确认，并优先复核失败 tuple</h2>
    <ol>
      <li>若主视图出现 hard-gate 下降，先逐条审阅 failure_samples，再决定是否进入后续阶段。</li>
      <li>若 hard gates 不下降但语义 CI 跨 0，将结论保持为“稳健但增量不确定”。</li>
      <li>新增从未参与 cp318 选点的 meetings，复用同一随机合同完成独立确认。</li>
    </ol>
    <h3>仍需回答的问题</h3>
    <ul><li>失败是否集中在特定 meeting、长度或数字/日期密度？</li><li>新增 meeting 上的语义增量能否在不牺牲 hard gates 的前提下复现？</li></ul>
  </section>

  <section id="provenance">
    <div class="section-kicker">可审计来源</div>
    <h2>报告绑定 score、draws、failures、results 与 greedy anchor</h2>
    <div class="table-wrap"><table><thead><tr><th>Artifact</th><th>File SHA-256</th><th>Payload SHA-256</th></tr></thead><tbody>{source_rows}</tbody></table></div>
  </section>
  <footer>Self-contained portable HTML；无外部字体、脚本、图片或网络依赖。权威数值以绑定的 sealed score artifacts 为准。</footer>
</main></body></html>
"""


def render_and_seal_report(
    *,
    score_manifest: Path,
    score_manifest_sha256: str,
    output_dir: Path,
    title: str,
) -> dict[str, Any]:
    """Validate a formal score bundle and publish report.html + manifest.json."""

    output = output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise BootstrapReportError(f"refusing to reuse report directory: {output}")
    source_binding = _file_binding(Path(__file__))
    bundle = load_and_validate_bundle(score_manifest, score_manifest_sha256)
    score = bundle["score"]
    artifact_bindings = {
        "score_manifest": bundle["score_binding"],
        "bootstrap_results": dict(score["artifacts"]["bootstrap_results"]),
        "row_scores": dict(score["artifacts"]["row_scores"]),
        "failure_samples": dict(score["artifacts"]["failure_samples"]),
        "bootstrap_draws": dict(score["artifacts"]["bootstrap_draws"]),
        "greedy_n12_anchor": bundle["greedy_binding"],
        "generation_suite": dict(score["inputs"]["generation_suite_manifest"]),
        "canonical_full_sample_manifest": dict(score["inputs"]["sample_manifest"]),
        "selection_n12_sample_manifest": dict(
            score["inputs"]["selection_n12_sample_manifest"]
        ),
        "semantic_model_manifest": dict(score["inputs"]["semantic_model_manifest"]),
    }
    report_html = render_report_html(
        score=score,
        results=bundle["results"],
        greedy=bundle["greedy"],
        failures=bundle["failures"],
        bindings=artifact_bindings,
        title=title,
    )
    required_fragments = (
        'id="full"',
        'id="row_disjoint"',
        'id="strict_meeting_disjoint"',
        'id="greedy-anchor"',
        'id="failures"',
        'id="method"',
        'id="limitations"',
        'id="provenance"',
        "post-selection",
    )
    if any(fragment not in report_html for fragment in required_fragments):
        raise BootstrapReportError("portable report is missing a required section")
    if (
        "http://" in report_html
        or "https://" in report_html
        or "<script" in report_html.casefold()
    ):
        raise BootstrapReportError("portable report unexpectedly references the network")
    output.mkdir(parents=True, exist_ok=False)
    _fsync_directory(output.parent)
    report_path = output / "report.html"
    _write_new_text(report_path, report_html)
    report_binding = {
        **_file_binding(report_path),
        "media_type": "text/html; charset=utf-8",
        "self_contained": True,
        "network_dependencies": False,
    }

    # Revalidate all score inputs before publishing a status=complete manifest.
    refreshed = load_and_validate_bundle(score_manifest, score_manifest_sha256)
    if refreshed["score_binding"] != bundle["score_binding"]:
        raise BootstrapReportError("score bundle changed during report rendering")
    if _file_binding(Path(source_binding["path"])) != source_binding:
        raise BootstrapReportError("report renderer changed during rendering")
    manifest = seal_manifest(
        {
            "schema_version": REPORT_MANIFEST_SCHEMA_VERSION,
            "status": "complete",
            "operation": "render_only_no_metric_or_bootstrap_recomputation",
            "created_at_utc": _utc_now(),
            "title": title,
            "report": report_binding,
            "inputs": artifact_bindings,
            "renderer": source_binding,
            "report_contract": {
                "audience": "technical",
                "portable_self_contained_html": True,
                "view_order": list(VIEW_ORDER),
                "metric_order": list(METRIC_ORDER),
                "greedy_anchor_displayed_separately": True,
                "failure_counts_and_examples_displayed": True,
                "post_selection_caveat_displayed": True,
                "weighted_composite_displayed": False,
                "structural_validation": {
                    "required_sections": [
                        "技术摘要",
                        "确定性 anchor",
                        "失败审计",
                        "实验与统计设计",
                        "不确定性与稳健性边界",
                        "可审计来源",
                    ],
                    "external_network_references": 0,
                },
            },
            "limitations": [
                "The HTML renders already-sealed values and does not independently recompute statistics.",
                "Portable delivery received structural validation; browser pixel rendering is environment-dependent.",
                "The report remains post-selection robustness and is not an independent confirmation set.",
            ],
        }
    )
    manifest_path = output / "manifest.json"
    _write_new_json(manifest_path, manifest)
    validate_manifest_integrity(manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-manifest", required=True, type=Path)
    parser.add_argument("--score-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--title",
        default="CHK0 / CHK1 / CHK3 cp318 随机生成 Bootstrap 评测",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = render_and_seal_report(
            score_manifest=args.score_manifest,
            score_manifest_sha256=args.score_manifest_sha256,
            output_dir=args.output_dir,
            title=args.title,
        )
    except (BootstrapReportError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        _canonical_json(
            {
                "status": manifest["status"],
                "report": manifest["report"]["path"],
                "report_sha256": manifest["report"]["sha256"],
                "payload_sha256": manifest["integrity"]["payload_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
