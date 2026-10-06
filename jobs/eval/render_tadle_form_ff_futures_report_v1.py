#!/usr/bin/env python3
"""Render a portable-report input and paper-facing tables for the Tadle-form diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE = (
    ROOT / "output/evaluation/main/tadle_form_official_generated_ff_futures_2004_2015_v1"
)
DEFAULT_VALIDATION = (
    ROOT
    / "output/evaluation/main/"
    "tadle_form_official_generated_ff_futures_2004_2015_v1_deep_validation_v3"
)
DEFAULT_OUTPUT = (
    ROOT
    / "output/evaluation/main/"
    "tadle_form_official_generated_ff_futures_2004_2015_v1_reporting_addendum_v2"
)
PRIMARY_BACKEND = "distilbert_fomc_9c061b4_v1"
ROBUSTNESS_BACKEND = "prosus_finbert_4556d13_v1"
PRIMARY_CONVENTION = "calendar_month_offset"
PRIMARY_SPECIFICATION = "basic_tadle_form"
ARMS = ("official", "chk0", "chk1", "chk3")
LABELS = {
    "official": "Official Minutes",
    "chk0": "Model chk-0",
    "chk1": "Model chk-1 cp200",
    "chk3": "Model chk-2 cp318",
}
SCHEMA = "tadle-form-ff-futures-reporting-addendum-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n")


def binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(relative_to.resolve())),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):+.{digits}f}"


def backend_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    backend_id: str = PRIMARY_BACKEND,
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if row["backend_id"] == backend_id
        and row["convention"] == PRIMARY_CONVENTION
        and row["specification"] == PRIMARY_SPECIFICATION
    ]


def coefficients_markdown(
    rows: list[dict[str, Any]],
    *,
    convention: str,
    backend_id: str = PRIMARY_BACKEND,
) -> str:
    selected = [
        row for row in rows
        if row["backend_id"] == backend_id
        and row["convention"] == convention
        and row["specification"] == PRIMARY_SPECIFICATION
    ]
    lookup = {(row["horizon_label"], row["arm"]): row for row in selected}
    lines = [
        "| Horizon | N | Official Minutes | Model chk-0 | Model chk-1 cp200 | Model chk-2 cp318 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for horizon in ("FF1", "FF3", "FF6", "FF12"):
        first = lookup[(horizon, "official")]
        cells = []
        for arm in ARMS:
            row = lookup[(horizon, arm)]
            cells.append(
                f"{fmt(row['beta_news_shock_basis_points'])} "
                f"({float(row['hc1_se_news_shock_basis_points']):.4f})"
            )
        lines.append(f"| {horizon} | {int(first['nobs']):,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def contrasts_markdown(
    rows: list[dict[str, Any]],
    *,
    backend_id: str = PRIMARY_BACKEND,
) -> str:
    selected = backend_rows(rows, backend_id=backend_id)
    lines = [
        "| Horizon | Model | Difference (bp) | 95% paired bootstrap CI | Holm p |",
        "|---|---|---:|---:|---:|",
    ]
    for row in sorted(selected, key=lambda item: (int(item["horizon_months"]), item["arm"])):
        lines.append(
            f"| {row['horizon_label']} | {LABELS[row['arm']]} | "
            f"{fmt(row['estimate_basis_points'])} | "
            f"[{fmt(row['ci_95_low_basis_points'])}, {fmt(row['ci_95_high_basis_points'])}] | "
            f"{float(row['holm_p']):.6f} |"
        )
    return "\n".join(lines)


def gains_markdown(
    rows: list[dict[str, Any]],
    *,
    backend_id: str = PRIMARY_BACKEND,
) -> str:
    selected = backend_rows(rows, backend_id=backend_id)
    lines = [
        "| Horizon | Comparator | chk-2 distance gain (bp) | 95% paired bootstrap CI | Holm p |",
        "|---|---|---:|---:|---:|",
    ]
    for row in sorted(selected, key=lambda item: (int(item["horizon_months"]), item["comparator_arm"])):
        lines.append(
            f"| {row['horizon_label']} | {LABELS[row['comparator_arm']]} | "
            f"{fmt(row['estimate_basis_points'])} | "
            f"[{fmt(row['ci_95_low_basis_points'])}, {fmt(row['ci_95_high_basis_points'])}] | "
            f"{float(row['holm_p']):.6f} |"
        )
    return "\n".join(lines)


def paper_table(coefficients: list[dict[str, Any]], contrasts: list[dict[str, Any]]) -> str:
    primary_coefficients = backend_rows(coefficients)
    coefficient_lookup = {
        (row["horizon_label"], row["arm"]): row for row in primary_coefficients
    }
    contrast_lookup = {
        (row["horizon_label"], row["arm"]): row for row in backend_rows(contrasts)
    }
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{3pt}",
        r"    \renewcommand{\arraystretch}{1.12}",
        r"    \caption{Tadle-Form Federal Funds Futures Coefficient Diagnostic}",
        r"    \label{tab:ch2:tadle_form_ff_futures}",
        r"    \begin{threeparttable}",
        r"    \begin{tabular}{lrrrrrr}",
        r"        \toprule",
        r"        Horizon & $N$ & Official & \textit{Model chk-0} & \textit{Model chk-1} & \textit{Model chk-2} & chk-2 $-$ Official \\",
        r"        \midrule",
    ]
    for horizon in ("FF1", "FF3", "FF6", "FF12"):
        official = coefficient_lookup[(horizon, "official")]
        cells = []
        for arm in ARMS:
            row = coefficient_lookup[(horizon, arm)]
            cells.append(
                rf"\makecell{{{fmt(row['beta_news_shock_basis_points'])}\\({float(row['hc1_se_news_shock_basis_points']):.4f})}}"
            )
        contrast = contrast_lookup[(horizon, "chk3")]
        lines.append(
            f"        {horizon} & {int(official['nobs']):,} & "
            + " & ".join(cells)
            + rf" & \makecell{{{fmt(contrast['estimate_basis_points'])}\\$p_{{\mathrm{{Holm}}}}={float(contrast['holm_p']):.3f}$}} \\"
        )
    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \begin{tablenotes}[flushleft]",
        r"        \footnotesize",
        r"        \item \textit{Notes:} The table reports the primary DistilBERT FOMC-stance results under the calendar-month-offset contract convention. Coefficients and HC1 standard errors (in parentheses) are in basis points. Model-minus-official inference uses 10,000 paired calendar-year block-bootstrap draws; Holm adjustment covers the twelve prespecified source--horizon contrasts. Generated documents were not historically released, so these are document-source association diagnostics rather than market effects.",
        r"    \end{tablenotes}",
        r"    \end{threeparttable}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def source(source_id: str, label: str, path: str, sql: str, description: str) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "description": description,
            "tables_used": [path],
        },
    }


def build_artifact(
    release: Path,
    validation: Path,
    coefficients: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    gains: list[dict[str, Any]],
) -> dict[str, Any]:
    base = str(release.resolve().relative_to(ROOT.resolve()))
    validation_path = str((validation / "validation_report.json").resolve().relative_to(ROOT.resolve()))
    coefficient_path = f"{base}/estimation/coefficient_estimates.jsonl"
    contrast_path = f"{base}/estimation/model_minus_official_contrasts.jsonl"
    gain_path = f"{base}/estimation/distance_gains.jsonl"
    summary_path = f"{base}/result_summary.json"
    technical_path = f"{base}/technical_report.md"
    sources = [
        source(
            "result_summary",
            "Frozen result summary",
            summary_path,
            f"SELECT * FROM read_json_auto('{summary_path}')",
            "Reads the frozen headline result counts.",
        ),
        source(
            "coefficient_estimates",
            "Frozen coefficient estimates",
            coefficient_path,
            f"SELECT * FROM read_json_auto('{coefficient_path}')",
            "Reads all basic, interaction, backend, horizon, and contract-convention coefficient estimates.",
        ),
        source(
            "model_contrasts",
            "Paired model-minus-official contrasts",
            contrast_path,
            f"SELECT * FROM read_json_auto('{contrast_path}')",
            "Reads the primary paired model-minus-official contrast family.",
        ),
        source(
            "distance_gains",
            "Paired coefficient-distance gains",
            gain_path,
            (
                "SELECT horizon_label, horizon_months, comparator_arm, estimate_basis_points, "
                "ci_95_low_basis_points, ci_95_high_basis_points, holm_p, bootstrap_draws "
                f"FROM read_json_auto('{gain_path}') "
                f"WHERE backend_id = '{PRIMARY_BACKEND}' "
                f"AND convention = '{PRIMARY_CONVENTION}' "
                f"AND specification = '{PRIMARY_SPECIFICATION}' "
                "ORDER BY horizon_months, comparator_arm"
            ),
            "Selects the primary Model chk-2 coefficient-distance-gain family.",
        ),
        source(
            "distance_gains_finbert",
            "FinBERT coefficient-distance gains",
            gain_path,
            (
                "SELECT horizon_label, horizon_months, comparator_arm, estimate_basis_points, "
                "ci_95_low_basis_points, ci_95_high_basis_points, holm_p, bootstrap_draws "
                f"FROM read_json_auto('{gain_path}') "
                f"WHERE backend_id = '{ROBUSTNESS_BACKEND}' "
                f"AND convention = '{PRIMARY_CONVENTION}' "
                f"AND specification = '{PRIMARY_SPECIFICATION}' "
                "ORDER BY horizon_months, comparator_arm"
            ),
            "Selects the non-pooled FinBERT Model chk-2 coefficient-distance-gain family.",
        ),
        source(
            "deep_validation",
            "Independent deep-validation receipt",
            validation_path,
            f"SELECT * FROM read_json_auto('{validation_path}')",
            "Reads the independent replay status and maximum numerical errors.",
        ),
        {
            "id": "technical_report",
            "label": "Frozen technical report",
            "path": technical_path,
        },
    ]
    primary_coefficients = backend_rows(coefficients)
    primary_contrasts = backend_rows(contrasts)
    primary_gains = backend_rows(gains)
    robustness_coefficients = backend_rows(coefficients, backend_id=ROBUSTNESS_BACKEND)
    robustness_contrasts = backend_rows(contrasts, backend_id=ROBUSTNESS_BACKEND)
    robustness_gains = backend_rows(gains, backend_id=ROBUSTNESS_BACKEND)
    chart_rows = [
        {
            "horizon_label": row["horizon_label"],
            "horizon_months": int(row["horizon_months"]),
            "comparator_arm": row["comparator_arm"],
            "comparator_label": LABELS[row["comparator_arm"]],
            "estimate_basis_points": float(row["estimate_basis_points"]),
            "ci_95_low_basis_points": float(row["ci_95_low_basis_points"]),
            "ci_95_high_basis_points": float(row["ci_95_high_basis_points"]),
            "holm_p": float(row["holm_p"]),
        }
        for row in sorted(primary_gains, key=lambda item: (int(item["horizon_months"]), item["comparator_arm"]))
    ]
    charts = [{
        "id": "primary_distance_gains",
        "title": "Model chk-2 coefficient-distance gains by futures horizon",
        "type": "bar",
        "dataset": "primary_distance_gains",
        "sourceId": "distance_gains",
        "encodings": {
            "x": {"field": "horizon_label", "type": "ordinal", "label": "Futures horizon"},
            "y": {
                "field": "estimate_basis_points",
                "type": "quantitative",
                "label": "Coefficient-distance gain",
                "unit": "bp",
            },
            "color": {"field": "comparator_label", "type": "nominal", "label": "Comparator"},
        },
    }]
    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# Tadle-Form Federal Funds Futures Document-Source Diagnostic",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "result_summary",
            "body": (
                "## Result in brief\n\n"
                "The primary specification finds no Holm-significant model-minus-official coefficient difference "
                "across 12 prespecified comparisons and no Holm-significant Model chk-2 coefficient-distance gain "
                "across eight comparisons. The test therefore does not establish either a difference from or "
                "equivalence to the official-Minutes coefficient."
            ),
        },
        {
            "id": "scope",
            "type": "markdown",
            "sourceId": "technical_report",
            "body": (
                "## Scope and estimand\n\n"
                "This is a Tadle-form adaptation on daily 30-Day Federal Funds futures settlements from "
                "December 1, 2004 through April 30, 2015. The analysis compares the historical return coefficient "
                "associated with sentiment recovered from official Minutes with coefficients constructed from "
                "Model chk-0, Model chk-1 cp200, and Model chk-2 cp318 documents. Generated documents were never "
                "released; the coefficients are document-source association diagnostics, not market effects."
            ),
        },
        {
            "id": "primary_coefficients",
            "type": "markdown",
            "sourceId": "coefficient_estimates",
            "body": (
                "## Primary DistilBERT coefficient estimates\n\n"
                + coefficients_markdown(coefficients, convention=PRIMARY_CONVENTION)
                + "\n\nParentheses contain HC1 heteroskedasticity-robust standard errors; all values are basis points."
            ),
        },
        {
            "id": "paired_contrasts",
            "type": "markdown",
            "sourceId": "model_contrasts",
            "body": "## Paired model-minus-official inference\n\n" + contrasts_markdown(contrasts),
        },
        {
            "id": "distance_gain_chart",
            "type": "chart",
            "chartId": "primary_distance_gains",
            "layout": "full",
        },
        {
            "id": "distance_gain_table",
            "type": "markdown",
            "sourceId": "distance_gains",
            "body": (
                "## Model chk-2 coefficient-distance diagnostic\n\n"
                + gains_markdown(gains)
                + "\n\nPositive gains favour Model chk-2. Every interval includes zero, so the pointwise rankings are not bootstrap-stable."
            ),
        },
        {
            "id": "finbert_coefficients",
            "type": "markdown",
            "sourceId": "coefficient_estimates",
            "body": (
                "## FinBERT financial-valence robustness\n\n"
                + coefficients_markdown(
                    coefficients,
                    convention=PRIMARY_CONVENTION,
                    backend_id=ROBUSTNESS_BACKEND,
                )
                + "\n\nFinBERT is a separate financial-valence construct and is not pooled with the primary FOMC-stance backend."
            ),
        },
        {
            "id": "finbert_contrasts",
            "type": "markdown",
            "sourceId": "model_contrasts",
            "body": (
                "## FinBERT paired model-minus-official inference\n\n"
                + contrasts_markdown(contrasts, backend_id=ROBUSTNESS_BACKEND)
            ),
        },
        {
            "id": "finbert_distance_gains",
            "type": "markdown",
            "sourceId": "distance_gains_finbert",
            "body": (
                "## FinBERT Model chk-2 distance gains\n\n"
                + gains_markdown(gains, backend_id=ROBUSTNESS_BACKEND)
                + "\n\nEvery FinBERT interval also includes zero; the robustness backend therefore does not establish a stable closeness gain."
            ),
        },
        {
            "id": "roll_robustness",
            "type": "markdown",
            "sourceId": "coefficient_estimates",
            "body": (
                "## Alternative contract-ranking robustness\n\n"
                + coefficients_markdown(coefficients, convention="live_contract_rank")
                + "\n\nThe live-contract-rank construction changes several point estimates materially, especially FF1, "
                "but does not yield a stable generated-source ordering. This sensitivity is consistent with the "
                "absence of Tadle's proprietary continuation-series rule."
            ),
        },
        {
            "id": "interaction_robustness",
            "type": "markdown",
            "sourceId": "coefficient_estimates",
            "body": (
                "## Post-August-8, 2011 interaction robustness\n\n"
                "None of the 64 interaction coefficients reaches the conventional two-sided 5% threshold under "
                "HC1 standard errors. Interaction and live-rank estimates were not included in the frozen paired "
                "bootstrap family and remain descriptive robustness diagnostics."
            ),
        },
        {
            "id": "validation",
            "type": "markdown",
            "sourceId": "deep_validation",
            "body": (
                "## Independent replay validation\n\n"
                "The deep validator rechecks the sealed manifests, official Statement cleaning and model forward "
                "passes, futures mappings and returns, all point estimates, every bootstrap draw coefficient, and "
                "the resulting confidence intervals and multiplicity adjustments. The validation status is passed."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "technical_report",
            "body": (
                "## Limitations and interpretation\n\n"
                "The exercise is not a literal replication of Tadle (2022): it substitutes frozen neural sentiment "
                "backends for the original dictionary, retains the full official Minutes body, uses a later WRDS "
                "Datastream vintage, and supplies explicit contract-selection rules because the proprietary FF1/FF3/"
                "FF6/FF12 continuation convention is unavailable. Separate within-source standardization also makes "
                "coefficient magnitudes normalization-dependent. Failure to reject a coefficient difference is not "
                "evidence of statistical equivalence."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Recommended next step\n\n"
                "Treat this result as a bounded historical coefficient diagnostic. A stronger replication would "
                "require the published dictionary/scoring implementation, the original vendor continuation series "
                "and vintage, and the exact removal of Statement text repeated inside official Minutes."
            ),
        },
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Tadle-Form Federal Funds Futures Document-Source Diagnostic",
            "charts": charts,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "datasets": {
                "primary_distance_gains": chart_rows,
                "primary_coefficients": primary_coefficients,
                "primary_model_minus_official_contrasts": primary_contrasts,
                "primary_distance_gain_rows": primary_gains,
                "finbert_coefficients": robustness_coefficients,
                "finbert_model_minus_official_contrasts": robustness_contrasts,
                "finbert_distance_gain_rows": robustness_gains,
            },
        },
        "sources": sources,
    }


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"create-only report output exists: {args.output_root}")
    validation_report = args.validation_root / "validation_report.json"
    if not validation_report.is_file():
        raise FileNotFoundError(f"deep validation report missing: {validation_report}")
    if json.loads(validation_report.read_text(encoding="utf-8"))["status"] != "passed":
        raise RuntimeError("deep validation did not pass")
    coefficients = read_jsonl(args.release_root / "estimation/coefficient_estimates.jsonl")
    contrasts = read_jsonl(args.release_root / "estimation/model_minus_official_contrasts.jsonl")
    gains = read_jsonl(args.release_root / "estimation/distance_gains.jsonl")
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_root.name}.staging.", dir=args.output_root.parent))
    completed = False
    try:
        write_json(
            staging / "artifact.json",
            build_artifact(args.release_root, args.validation_root, coefficients, contrasts, gains),
        )
        write_text(staging / "paper_table.tex", paper_table(coefficients, contrasts))
        write_text(
            staging / "robustness_tables.md",
            "# Tadle-Form Robustness Tables\n\n"
            "## Live-contract-rank: DistilBERT FOMC stance\n\n"
            + coefficients_markdown(coefficients, convention="live_contract_rank")
            + "\n",
        )
        manifest = {
            "schema_version": SCHEMA + ":manifest",
            "status": "complete",
            "created_at_utc": utc_now(),
            "source_release": {
                "path": str(args.release_root.resolve()),
                "manifest_sha256": sha256_file(args.release_root / "manifest.json"),
            },
            "deep_validation": {
                "path": str(args.validation_root.resolve()),
                "manifest_sha256": sha256_file(args.validation_root / "manifest.json"),
            },
            "renderer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "artifacts": {
                "artifact_json": binding(staging / "artifact.json", relative_to=staging),
                "paper_table": binding(staging / "paper_table.tex", relative_to=staging),
                "robustness_tables": binding(staging / "robustness_tables.md", relative_to=staging),
            },
            "portable_html": "generated subsequently from artifact.json by the validated Data Analytics portable-report builder",
        }
        write_json(staging / "manifest.json", manifest)
        os.replace(staging, args.output_root)
        completed = True
    finally:
        if not completed and staging.exists():
            shutil.rmtree(staging)
    print(canonical({"status": "complete", "output_root": str(args.output_root.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
