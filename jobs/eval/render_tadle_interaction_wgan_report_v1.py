#!/usr/bin/env python3
"""Build a canonical technical-report artifact for the WGAN/Tadle diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE = (
    ROOT
    / "output/evaluation/main/"
    "tadle_interaction_wgan_dictionary_ff_futures_2004_2015_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "output/evaluation/main/"
    "tadle_interaction_wgan_dictionary_ff_futures_2004_2015_v1_report"
)
SCHEMA = "tadle-interaction-wgan-dictionary-report-v1"
ARMS = ("official", "chk0", "chk1", "chk3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def source(source_id: str, label: str, path: str, description: str) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_json_auto('{path}')",
            "description": description,
            "tables_used": [path],
        },
    }


def coefficient_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lookup = {(row["horizon_label"], row["arm"]): row for row in rows}
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
                f"{fmt(row['beta_pre_basis_points'])} ({row['hc1_se_pre_basis_points']:.4f}) / "
                f"{fmt(row['beta_interaction_basis_points'])} ({row['hc1_se_interaction_basis_points']:.4f}) / "
                f"{fmt(row['beta_post_basis_points'])} ({row['hc1_se_post_basis_points']:.4f})"
            )
        lines.append(f"| {horizon} | {first['nobs']:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def contrast_table(rows: Sequence[Mapping[str, Any]], *, arm: str = "chk3") -> str:
    lines = [
        "| Estimand | Horizon | Difference (bp) | Paired 95% CI | Holm p |",
        "|---|---|---:|---:|---:|",
    ]
    order = {"pre": 0, "interaction": 1, "post": 2}
    for row in sorted(
        [item for item in rows if item["arm"] == arm],
        key=lambda item: (order[item["estimand"]], item["horizon_months"]),
    ):
        lines.append(
            f"| {row['estimand']} | {row['horizon_label']} | {fmt(row['estimate_basis_points'])} | "
            f"[{fmt(row['ci_95_low_basis_points'])}, {fmt(row['ci_95_high_basis_points'])}] | "
            f"{row['holm_p']:.6f} |"
        )
    return "\n".join(lines)


def paper_table(points: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]]) -> str:
    lookup = {(row["horizon_label"], row["arm"]): row for row in points}
    post = {
        (row["horizon_label"], row["arm"]): row
        for row in contrasts if row["estimand"] == "post"
    }

    def stars(p_value: float) -> str:
        if p_value < 0.001:
            return "^{***}"
        if p_value < 0.01:
            return "^{**}"
        if p_value < 0.05:
            return "^{*}"
        return ""

    def distance_interval(low: float, high: float) -> tuple[float, float]:
        """Map a paired signed-contrast interval to the absolute-distance scale."""

        if low <= 0.0 <= high:
            return 0.0, max(abs(low), abs(high))
        return min(abs(low), abs(high)), max(abs(low), abs(high))

    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{1.5pt}",
        r"    \renewcommand{\arraystretch}{1.12}",
        r"    \caption{Tadle-Interaction Federal Funds Futures Coefficients and Distances to Official Minutes}",
        r"    \label{tab:ch2:tadle_wgan_interaction}",
        r"    \begin{threeparttable}",
        r"    \begin{tabularx}{\textwidth}{@{}l*{7}{>{\centering\arraybackslash}X}@{}}",
        r"        \toprule",
        r"        Horizon",
        r"        & \makecell{Official\\Minutes}",
        r"        & \makecell{\textit{Model chk-0}}",
        r"        & \makecell{\textit{Model chk-1}\\cp200}",
        r"        & \makecell{\textit{Model chk-2}\\cp318}",
        r"        & \makecell{$D_{0,O}$\\chk-0 vs. Official}",
        r"        & \makecell{$D_{1,O}$\\chk-1 vs. Official}",
        r"        & \makecell{$D_{2,O}$\\chk-2 vs. Official} \\",
        r"        \midrule",
    ]
    for horizon in ("FF1", "FF3", "FF6", "FF12"):
        cells = []
        for arm in ARMS:
            row = lookup[(horizon, arm)]
            cells.append(
                rf"\makecell{{{fmt(row['beta_post_basis_points'])}\\({row['hc1_se_post_basis_points']:.4f})}}"
            )
        contrast_cells = []
        for arm in ("chk0", "chk1", "chk3"):
            contrast = post[(horizon, arm)]
            distance_low, distance_high = distance_interval(
                float(contrast["ci_95_low_basis_points"]),
                float(contrast["ci_95_high_basis_points"]),
            )
            contrast_cells.append(
                rf"\makecell{{${abs(float(contrast['estimate_basis_points'])):.4f}{stars(contrast['holm_p'])}$\\"
                rf"$[{distance_low:.4f},\,{distance_high:.4f}]$}}"
            )
        lines.append(
            f"        {horizon} & " + " & ".join(cells)
            + " & " + " & ".join(contrast_cells) + r" \\"
        )
    lines.extend([
        r"        \bottomrule",
        r"    \end{tabularx}",
        r"    \begin{tablenotes}[flushleft]",
        r"        \footnotesize",
        r"        \item \textit{Notes:} The four source columns report the post-August-8-2011 marginal coefficient $\widehat\beta_{NS}+\widehat\beta_{post\times NS}$ in basis points, with HC1 standard errors in parentheses. The distance columns are $D_{n,O}=|\widehat\beta_{\mathrm{chk}\text{-}n}-\widehat\beta_{O}|$. Bracketed intervals are the absolute-value images of the paired 95\% signed-contrast intervals: if a signed interval contains zero, its distance interval begins at zero. Inference uses 10,000 paired, regime-stratified, 2011-anchored calendar-year block-bootstrap draws. Stars refer to the paired signed-contrast tests, with Holm adjustment jointly across the twelve post-period comparisons: $^{*}p_{\mathrm{Holm}}<0.05$, $^{**}p_{\mathrm{Holm}}<0.01$, and $^{***}p_{\mathrm{Holm}}<0.001$. The WGAN Loughran--McDonald polarity dictionary is not Tadle's original custom monetary-policy dictionary.",
        r"    \end{tablenotes}",
        r"    \end{threeparttable}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def build_artifact(
    release: Path, points: list[dict[str, Any]], contrasts: list[dict[str, Any]],
    release_manifest: Mapping[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    base = str(release.resolve().relative_to(ROOT.resolve()))
    coefficient_path = f"{base}/estimation/coefficient_estimates.jsonl"
    contrast_path = f"{base}/estimation/model_minus_official_contrasts.jsonl"
    score_path = f"{base}/sentiment/document_scores.jsonl"
    removal_path = f"{base}/sentiment/official_statement_removal_ledger.jsonl"
    report_path = f"{base}/technical_report.md"
    validation_path = f"{base}/validation_report.json"
    sources = [
        source("coefficients", "Full-sample interaction coefficients", coefficient_path, "Reads the 16 frozen source-by-horizon coefficient rows."),
        source("contrasts", "Paired bootstrap coefficient contrasts", contrast_path, "Reads 36 model-minus-official contrast rows and Holm-adjusted p-values."),
        source("document_scores", "WGAN dictionary document scores", score_path, "Reads the bound WGAN LM polarity scores for official and generated documents."),
        source("statement_removal", "Official Minutes Statement-removal audit", removal_path, "Reads the 84-row passage-selection and removal audit."),
        source("technical_report", "Frozen technical report", report_path, "Reads the English methods and interpretation report."),
        source("validation", "Frozen validation report", validation_path, "Reads release validation checks and limitations."),
    ]
    post_rows = [
        {
            "horizon_label": row["horizon_label"],
            "horizon_months": row["horizon_months"],
            "arm": row["arm"],
            "paper_label": row["paper_label"],
            "beta_post_basis_points": row["beta_post_basis_points"],
            "hc1_se_post_basis_points": row["hc1_se_post_basis_points"],
            "nobs": row["nobs"],
        }
        for row in points
    ]
    primitive_significant = sum(
        row["holm_p"] < 0.05 and row["family"] == "primitive_pre_and_interaction_24"
        for row in contrasts
    )
    post_significant = sum(
        row["holm_p"] < 0.05 and row["family"] == "derived_post_marginal_12"
        for row in contrasts
    )
    chart = {
        "id": "post_marginal_coefficients",
        "title": "Post-2011 marginal news-shock coefficients by document source",
        "type": "bar",
        "dataset": "post_marginal_coefficients",
        "sourceId": "coefficients",
        "encodings": {
            "x": {"field": "horizon_label", "label": "Federal Funds futures horizon", "type": "ordinal"},
            "y": {"field": "beta_post_basis_points", "label": "Post-2011 marginal coefficient", "type": "quantitative", "unit": "bp"},
            "color": {"field": "paper_label", "label": "Document source", "type": "nominal"},
        },
    }
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Tadle-Interaction Federal Funds Futures Diagnostic Using the WGAN Dictionary"},
        {
            "id": "answer",
            "type": "markdown",
            "sourceId": "contrasts",
            "body": (
                "## Result in brief\n\n"
                "The requested interaction-form calculation is complete. None of the 24 primitive baseline/interaction "
                f"contrasts survives Holm correction (observed count: {primitive_significant}). In the separate, "
                f"secondary family, {post_significant} of 12 derived post-2011 marginal-slope contrasts are significant: "
                "all three generated sources differ from official Minutes at FF3, FF6, and FF12. The result is "
                "conditional on the frozen adaptation and does not establish causal market effects."
            ),
        },
        {
            "id": "scope",
            "type": "markdown",
            "sourceId": "technical_report",
            "body": (
                "## Scope and estimand\n\nThe equation matches Tadle's Federal Funds futures interaction form: "
                "calendar-year fixed effects and VIX are included, the post regime begins after August 8, 2011, "
                "and the post-period slope is the sum of the baseline and interaction coefficients. The WGAN "
                "dictionary and project market panel make this an adaptation rather than a literal replication."
            ),
        },
        {
            "id": "coefficient_table",
            "type": "markdown",
            "sourceId": "coefficients",
            "body": (
                "## Full-sample OLS coefficients\n\nValues are basis points; HC1 standard errors are in parentheses. "
                "Each cell reports baseline / interaction / post-period marginal.\n\n" + coefficient_table(points)
            ),
        },
        {"id": "chart", "type": "chart", "chartId": "post_marginal_coefficients", "layout": "full"},
        {
            "id": "chk2_contrasts",
            "type": "markdown",
            "sourceId": "contrasts",
            "body": (
                "## Model chk-2 minus official Minutes\n\nBootstrap intervals and p-values are computed from "
                "drawwise coefficient differences, not from differences between separate interval endpoints.\n\n"
                + contrast_table(contrasts)
            ),
        },
        {
            "id": "measurement",
            "type": "markdown",
            "sourceId": "document_scores",
            "body": (
                "## WGAN dictionary measurement\n\nThe score is `(positive-negative)/max(1, positive+negative)` "
                "under the historical WGAN tokenizer. The bound LM file contributes 347 positive and 2,345 "
                "negative unique lexemes. This measures financial valence, not Tadle's original hawkish/dovish construct."
            ),
        },
        {
            "id": "cleaning",
            "type": "markdown",
            "sourceId": "statement_removal",
            "body": (
                "## Official-text preprocessing\n\nThe repeated current-meeting Statement passage is removed from each "
                "official Minutes document before scoring. All 84 current/lag documents pass the frozen candidate-match "
                "gate; meetings with multiple policy-statement passages are resolved against the same-day Statement."
            ),
        },
        {
            "id": "bootstrap",
            "type": "markdown",
            "sourceId": "technical_report",
            "body": (
                "## Inference\n\nFull-sample OLS supplies the point estimates. A final 10,000-draw paired bootstrap "
                "supplies confidence intervals and p-values. The design resamples pre- and post-regime calendar-year "
                "blocks separately, always retains 2011, and resamples the five aligned generated documents within each "
                "meeting. Conditioning on the single 2011 bridge year can understate bridge-year uncertainty."
            ),
        },
        {
            "id": "interpretation",
            "type": "markdown",
            "sourceId": "coefficients",
            "body": (
                "## Interpretation\n\nThe official post-period slope is negative at every horizon and is largest at "
                "FF12. The generated-source post-period slopes are generally much nearer zero. The paired bootstrap "
                "detects positive model-minus-official differences for every generated source at FF3, FF6, and FF12; "
                "the corresponding FF1 differences are not significant. This is conditional evidence about a secondary "
                "derived slope family, not proof that generated documents are equivalent substitutes for official Minutes."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "validation",
            "body": (
                "## Limitations\n\nGenerated texts were never released and therefore could not have caused historical "
                "futures returns. The WGAN dictionary differs from Tadle's custom lexicon, Core8 documents are narrower "
                "than official Minutes, and the WRDS continuation rule is project-specific. Treat all coefficients as "
                "historical document-source association diagnostics."
            ),
        },
        {
            "id": "reproducibility",
            "type": "markdown",
            "sourceId": "validation",
            "body": (
                "## Reproducibility\n\nThe release binds the scorer, dictionary, source documents, market panel, "
                "standardization scales, 10,000-draw plan, coefficient ledger, and contrast ledger by SHA-256. "
                f"Release validation status: `{validation['status']}`."
            ),
        },
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Tadle-Interaction Federal Funds Futures Diagnostic Using the WGAN Dictionary",
            "blocks": blocks,
            "charts": [chart],
            "sources": sources,
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "datasets": {
                "post_marginal_coefficients": post_rows,
                "coefficient_estimates": points,
                "model_minus_official_contrasts": contrasts,
            },
        },
        "sources": sources,
    }


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"create-only output exists: {args.output_root}")
    manifest_path = args.release_root / "manifest.json"
    validation_path = args.release_root / "validation_report.json"
    points_path = args.release_root / "estimation/coefficient_estimates.jsonl"
    contrasts_path = args.release_root / "estimation/model_minus_official_contrasts.jsonl"
    release_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    points = read_jsonl(points_path)
    contrasts = read_jsonl(contrasts_path)
    if release_manifest["status"] != "complete" or validation["status"] != "passed":
        raise RuntimeError("source release is not complete and validated")
    if len(points) != 16 or len(contrasts) != 36:
        raise RuntimeError("source result inventory drift")
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_root.name}.staging.", dir=args.output_root.parent))
    completed = False
    try:
        write_json(
            staging / "artifact.json",
            build_artifact(args.release_root, points, contrasts, release_manifest, validation),
        )
        write_text(staging / "paper_table.tex", paper_table(points, contrasts))
        write_json(staging / "manifest.json", {
            "schema_version": SCHEMA + ":manifest",
            "created_at_utc": utc_now(),
            "status": "complete",
            "source_release": binding(manifest_path, relative_to=ROOT),
            "source_validation": binding(validation_path, relative_to=ROOT),
            "artifacts": {
                "artifact_json": binding(staging / "artifact.json", relative_to=staging),
                "paper_table": binding(staging / "paper_table.tex", relative_to=staging),
            },
            "portable_html": "generated subsequently from artifact.json by the validated Data Analytics builder",
        })
        os.replace(staging, args.output_root)
        completed = True
        print(canonical({"status": "complete", "output_root": str(args.output_root.resolve())}))
        return 0
    finally:
        if not completed and staging.exists():
            print(f"Partial staging retained for audit: {staging}", file=os.sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
