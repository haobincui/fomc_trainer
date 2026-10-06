#!/usr/bin/env python3
"""Run a create-only Tadle-form Federal Funds futures coefficient diagnostic.

This analysis adapts the equations documented in Tadle's author-primary
dissertation precursor to compare official FOMC Minutes with three generated
document sources.  It is not a literal replication: the project uses frozen
neural sentiment backends, WRDS Datastream settlement prices, and an explicit
calendar-month contract convention.  Generated documents were never released,
so their coefficients are historical document-source diagnostics rather than
market effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from jobs.eval import chk3_beta_statistics as beta_statistics
from jobs.eval import run_official_minutes_treasury_beta_v1 as official_runner


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = Path(__file__).resolve()
OFFICIAL_BENCHMARK = (
    ROOT / "output/evaluation/main/official_minutes_full_document_treasury_beta_1993_2025_v1"
)
MARKET_ROOT = (
    ROOT
    / "output/evaluation/main/chk3_beta_core8_merged_n2048_vllm_k5_20260816_v1"
    / "market_panel_fed_minutes_release_daily_v1"
)
WRDS_ROOT = (
    ROOT
    / "output/data/external/wrds_trdstrm_30day_fed_funds_contracts_20041129_20150430_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "output/evaluation/main/tadle_form_official_generated_ff_futures_2004_2015_v1"
)

RELEASE_LEDGER = MARKET_ROOT / "official_minutes_release_dates.v1.jsonl"
MARKET_PANEL = MARKET_ROOT / "market_panel.jsonl"
VIX_DAILY = MARKET_ROOT / "sources/market/VIXCLS.csv"
WRDS_METADATA = WRDS_ROOT / "contract_info.csv.gz"
WRDS_PRICES = WRDS_ROOT / "contract_prices_daily.csv.gz"
WRDS_MANIFEST = WRDS_ROOT / "manifest.json"
SAMPLE_START = "2004-12-01"
SAMPLE_END = "2015-04-30"
POST_2011_CUTOFF = "2011-08-08"
TADLE_GAMMA = 0.368
HORIZONS = (1, 3, 6, 12)
CONVENTIONS = ("calendar_month_offset", "live_contract_rank")
PRIMARY_CONVENTION = "calendar_month_offset"
ARMS = ("official", "chk0", "chk1", "chk3")
GENERATED_ARMS = ("chk0", "chk1", "chk3")
PAPER_LABELS = official_runner.PAPER_LABELS
BACKEND_IDS = (official_runner.DISTIL_ID, official_runner.FINBERT_ID)
SCHEMA = "tadle-form-ff-futures-document-source-diagnostic-v1"
STATEMENT_OVERRIDE = {
    # The official 2007 calendar snapshot omits this link.  The Federal Reserve
    # page is live but carries an anomalous 20070618 slug for a 2007-06-28 release.
    "2007-06-28": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20070618a.htm",
}
LAG_GAP_CURRENT = {
    # The frozen 256-meeting model roster excludes the regular 2009-11-03/04
    # meeting.  The next roster meeting may not skip that real communication
    # when constructing RZ[t-1], so its Minutes-release day is excluded.
    "2009-12-16": "2009-11-04",
}
MISSING_MODEL_RELEASE_DATES = {
    # Minutes of the excluded 2009-11-03/04 meeting were released on this day.
    # Its document-source shock is unavailable in every model arm.
    "2009-11-24": "frozen model roster omits the 2009-11-03/04 regular meeting",
}


class DiagnosticError(RuntimeError):
    """A frozen-input, construction, or publication contract failed closed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_824)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise DiagnosticError(f"non-object JSONL row: {path}")
    return rows


def write_new_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def write_new_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    write_new_text(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n")


def write_new_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(dict(row)) + "\n")
            count += 1
    return count


def file_binding(path: Path, *, relative_to: Path | None = None, rows: int | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise DiagnosticError(f"bound file missing or symlinked: {resolved}")
    shown: Path | str = resolved
    if relative_to is not None:
        shown = resolved.relative_to(relative_to.resolve())
    result: dict[str, Any] = {
        "path": str(shown),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def runtime_contract() -> dict[str, Any]:
    import torch
    import transformers

    expected_env = {
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    observed = {key: os.environ.get(key) for key in expected_env}
    if observed != expected_env:
        raise DiagnosticError(f"runtime environment drift: {observed}")
    if platform.python_version() != "3.11.5" or np.__version__ != "1.24.3":
        raise DiagnosticError(
            "runtime must be Python 3.11.5 / NumPy 1.24.3; "
            f"observed {platform.python_version()} / {np.__version__}"
        )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "environment": observed,
    }


def load_sample() -> dict[str, Any]:
    release_rows = sorted(read_jsonl(RELEASE_LEDGER), key=lambda row: str(row["meeting_end_date"]))
    current = [
        row for row in release_rows
        if SAMPLE_START <= str(row["release_date"]) <= SAMPLE_END
    ]
    if len(current) != 83:
        raise DiagnosticError(f"Tadle-form release sample drift: observed {len(current)}, expected 83")
    chronology = [str(row["generation_meeting_id"]) for row in release_rows]
    prior = {now: before for before, now in zip(chronology[:-1], chronology[1:], strict=True)}
    if any(str(row["generation_meeting_id"]) not in prior for row in current):
        raise DiagnosticError("a current sample meeting lacks an immediate prior meeting")
    estimation_current = [
        row for row in current if str(row["generation_meeting_id"]) not in LAG_GAP_CURRENT
    ]
    if len(estimation_current) != 82:
        raise DiagnosticError("expected one roster-adjacency exclusion from the 83 release events")
    lag_ids = [prior[str(row["generation_meeting_id"])] for row in estimation_current]
    needed_ids = tuple(dict.fromkeys(
        [str(row["generation_meeting_id"]) for row in current] + lag_ids
    ))
    by_id = {str(row["generation_meeting_id"]): row for row in release_rows}
    if len(needed_ids) != 84 or any(mid not in by_id for mid in needed_ids):
        raise DiagnosticError("expected 84 current-plus-lag meeting identities")
    return {
        "release_rows": release_rows,
        "current": current,
        "estimation_current": estimation_current,
        "prior": prior,
        "needed_ids": needed_ids,
        "by_id": by_id,
        "lag_gap_exclusions": [
            {
                "meeting_id": mid,
                "required_immediate_prior_meeting": required,
                "reason": "required prior regular meeting is absent from the frozen model-output roster",
            }
            for mid, required in sorted(LAG_GAP_CURRENT.items())
        ],
    }


def extract_statement_urls(sample: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    from bs4 import BeautifulSoup

    links: dict[str, list[dict[str, str]]] = defaultdict(list)
    calendar_paths = sorted({Path(str(sample["by_id"][mid]["source_calendar_path"])) for mid in sample["needed_ids"]})
    for path in calendar_paths:
        if not path.is_file():
            raise DiagnosticError(f"bound Federal Reserve calendar missing: {path}")
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        for anchor in soup.find_all("a"):
            if anchor.get_text(" ", strip=True).casefold() != "statement":
                continue
            href = str(anchor.get("href") or "")
            dates = re.findall(r"(?<!\d)(20\d{6})(?!\d)", href)
            if not dates:
                continue
            token = dates[-1]
            date = f"{token[:4]}-{token[4:6]}-{token[6:]}"
            links[date].append({
                "href": href,
                "url": urljoin("https://www.federalreserve.gov/", href),
                "calendar_path": str(path.resolve()),
                "calendar_sha256": sha256_file(path),
            })
    result: dict[str, dict[str, Any]] = {}
    for mid in sample["needed_ids"]:
        candidates = {row["url"]: row for row in links.get(mid, [])}
        if mid in STATEMENT_OVERRIDE:
            override = STATEMENT_OVERRIDE[mid]
            calendar_path = Path(str(sample["by_id"][mid]["source_calendar_path"]))
            candidates = {override: {
                "href": override,
                "url": override,
                "calendar_path": str(calendar_path.resolve()),
                "calendar_sha256": sha256_file(calendar_path),
            }}
        if len(candidates) != 1:
            raise DiagnosticError(f"Statement URL is not unique for {mid}: {sorted(candidates)}")
        result[mid] = next(iter(candidates.values()))
    return result


def clean_statement_html(raw: bytes) -> tuple[str, dict[str, Any]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "html.parser")
    article = soup.select_one("#article")
    selector: str
    if article is not None:
        candidates = [
            node for node in article.select("div.col-xs-12.col-sm-8.col-md-8")
            if "committee" in node.get_text(" ", strip=True).casefold()
            and "heading" not in (node.get("class") or [])
        ]
        selected = max(candidates, key=lambda node: len(node.get_text(" ", strip=True))) if candidates else article
        selector = "#article substantive column" if candidates else "#article"
    else:
        candidates = [
            node for node in soup.find_all("td")
            if "committee" in node.get_text(" ", strip=True).casefold()
        ]
        selected = max(candidates, key=lambda node: len(node.get_text(" ", strip=True))) if candidates else soup.body
        selector = "largest Committee-bearing td" if candidates else "body"
    if selected is None:
        raise DiagnosticError("no supported Statement body container")
    fragment = BeautifulSoup(str(selected), "html.parser")
    removed = 0
    for css in (
        "script", "style", "noscript", "nav", "header", "footer", "form", "button", "svg", "img",
        ".heading", ".shareDL", "#lastUpdate", ".lastUpdate", ".footer", "[role='navigation']",
    ):
        for node in fragment.select(css):
            node.decompose()
            removed += 1
    text = re.sub(r"\s+", " ", fragment.get_text(" ", strip=True)).strip()
    immediate = re.search(r"\bFor immediate release\b", text, flags=re.IGNORECASE)
    if immediate and immediate.end() < min(500, len(text)):
        text = text[immediate.end():].strip(" :-")
    for footer in ("Last Update:", "Last update:", "Home | News and events", "Home | Monetary policy"):
        position = text.find(footer)
        if position >= 0:
            text = text[:position].strip()
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 200 or "committee" not in text.casefold():
        raise DiagnosticError(
            f"cleaned Statement failed content gate: selector={selector}, characters={len(text)}"
        )
    return text, {
        "selector": selector,
        "removed_nodes": removed,
        "characters": len(text),
        "whitespace_words": len(text.split()),
        "text_sha256": sha256_text(text),
        "no_silent_truncation": True,
    }


def fetch_statements(staging: Path, sample: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    mappings = extract_statement_urls(sample)
    session = requests.Session()
    retry = Retry(total=5, connect=5, read=5, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({
        "User-Agent": "FOMC academic reproducibility diagnostic/1.0 (official Statement archival fetch)"
    })
    source_rows: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for mid in sample["needed_ids"]:
        mapping = mappings[mid]
        response = session.get(mapping["url"], timeout=60, allow_redirects=True)
        response.raise_for_status()
        raw = response.content
        if len(raw) < 500 or "html" not in response.headers.get("Content-Type", "").casefold():
            raise DiagnosticError(f"Statement response failed HTML/size gate: {mid}")
        raw_path = staging / "statements/source/raw" / f"statement-{mid}.html"
        write_new_bytes(raw_path, raw)
        text, clean = clean_statement_html(raw)
        source_rows.append({
            "schema_version": SCHEMA + ":statement-source-row",
            "meeting_id": mid,
            "requested_url": mapping["url"],
            "final_url": response.url,
            "http_status": int(response.status_code),
            "content_type": response.headers.get("Content-Type"),
            "raw_path": str(raw_path.relative_to(staging)),
            "raw_bytes": len(raw),
            "raw_sha256": sha256_bytes(raw),
            "calendar_path": mapping["calendar_path"],
            "calendar_sha256": mapping["calendar_sha256"],
            "manual_official_url_override": mid in STATEMENT_OVERRIDE,
            **clean,
        })
        documents.append({
            "schema_version": SCHEMA + ":statement-document-row",
            "document_id": f"statement::{mid}",
            "meeting_id": mid,
            "meeting_end_date": mid,
            "document_text": text,
            "document_text_sha256": clean["text_sha256"],
            "source_raw_sha256": sha256_bytes(raw),
            "no_silent_truncation": True,
        })
    if len(source_rows) != 84 or len({row["meeting_id"] for row in source_rows}) != 84:
        raise DiagnosticError("Statement source coverage is not 84 unique meetings")
    write_new_jsonl(staging / "statements/source_manifest.jsonl", source_rows)
    write_new_jsonl(staging / "statements/documents.jsonl", documents)
    return source_rows, documents


def score_statement_backend(
    staging: Path,
    documents: Sequence[Mapping[str, Any]],
    backend_id: str,
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = official_runner.BACKENDS[backend_id]
    model_manifest_path = model_dir / "model_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    signed = model_manifest["signed_score"]
    output_dir = staging / "statements/sentiment" / backend_id
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    if tokenizer.cls_token_id != 101 or tokenizer.sep_token_id != 102:
        raise DiagnosticError(f"special-token contract drift: {backend_id}")
    torch.manual_seed(0)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise DiagnosticError("CUDA requested but unavailable")
        torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval().to(device)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    accumulators: dict[str, dict[str, Any]] = defaultdict(lambda: {"weight": 0.0, "probs": [0.0, 0.0, 0.0]})
    meta: dict[str, dict[str, Any]] = {}
    pending: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    window_rows: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        padded = tokenizer.pad(
            {"input_ids": [row[1]["input_ids"] for row in pending]},
            padding=True,
            return_tensors="pt",
        )
        padded = {key: value.to(device) for key, value in padded.items()}
        with torch.inference_mode():
            probs_batch = torch.softmax(model(**padded).logits.float(), dim=-1).cpu().tolist()
        for (document, spec), probs in zip(pending, probs_batch, strict=True):
            if len(probs) != 3 or not math.isclose(sum(probs), 1.0, abs_tol=2e-6):
                raise DiagnosticError("Statement classifier probability contract drift")
            doc_id = str(document["document_id"])
            weight = float(spec["aggregation_weight"])
            accumulators[doc_id]["weight"] += weight
            for index, probability in enumerate(probs):
                accumulators[doc_id]["probs"][index] += weight * float(probability)
            window_rows.append({
                "schema_version": SCHEMA + ":statement-window-score-row",
                "backend_id": backend_id,
                "document_id": doc_id,
                "meeting_id": document["meeting_id"],
                "window_index": spec["window_index"],
                "token_start": spec["token_start"],
                "token_end": spec["token_end"],
                "content_token_count": int(spec["token_end"] - spec["token_start"]),
                "content_token_ids_sha256": spec["content_token_ids_sha256"],
                "aggregation_weight": weight,
                "probabilities": [float(value) for value in probs],
                "score": float(probs[int(signed["positive_index"])]) - float(probs[int(signed["negative_index"])]),
            })
        pending = []

    for document in documents:
        specs = official_runner.window_specs(tokenizer, str(document["document_text"]))
        meta[str(document["document_id"])] = {
            "window_count": len(specs),
            "content_token_count": int(specs[-1]["token_end"]),
            "coverage_corrected_weight": float(sum(row["aggregation_weight"] for row in specs)),
        }
        for spec in specs:
            pending.append((document, spec))
            if len(pending) >= batch_size:
                flush()
    flush()

    score_rows: list[dict[str, Any]] = []
    for document in documents:
        doc_id = str(document["document_id"])
        acc = accumulators[doc_id]
        doc_meta = meta[doc_id]
        if not math.isclose(acc["weight"], doc_meta["content_token_count"], abs_tol=1e-7):
            raise DiagnosticError(f"Statement aggregation weight drift: {doc_id}")
        probs = [float(value) / acc["weight"] for value in acc["probs"]]
        score_rows.append({
            "schema_version": SCHEMA + ":statement-document-score-row",
            "backend_id": backend_id,
            "construct": model_manifest["construct"],
            "document_id": doc_id,
            "meeting_id": document["meeting_id"],
            "document_text_sha256": document["document_text_sha256"],
            **doc_meta,
            "label_order": model_manifest["label_order"],
            "probabilities": probs,
            "score": float(probs[int(signed["positive_index"])]) - float(probs[int(signed["negative_index"])]),
            "no_silent_truncation": True,
        })
    window_path = output_dir / "window_scores.jsonl"
    score_path = output_dir / "document_scores.jsonl"
    write_new_jsonl(window_path, window_rows)
    write_new_jsonl(score_path, score_rows)
    manifest = {
        "schema_version": SCHEMA + ":statement-sentiment-manifest",
        "created_at_utc": utc_now(),
        "backend_id": backend_id,
        "construct": model_manifest["construct"],
        "model_manifest": file_binding(model_manifest_path),
        "window_contract": {
            "content_tokens": official_runner.CONTENT_TOKENS,
            "overlap_tokens": official_runner.OVERLAP_TOKENS,
            "step": official_runner.WINDOW_STEP,
            "aggregation": "coverage-corrected probability mean",
            "signed_score": signed,
        },
        "coverage": {
            "documents": len(score_rows),
            "windows": len(window_rows),
            "empty_documents": 0,
            "silent_truncations": 0,
        },
        "artifacts": {
            "window_scores": file_binding(window_path, relative_to=staging, rows=len(window_rows)),
            "document_scores": file_binding(score_path, relative_to=staging, rows=len(score_rows)),
        },
        "runtime": runtime_contract(),
    }
    write_new_json(output_dir / "manifest.json", manifest)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return manifest


def add_calendar_month(date: pd.Timestamp, offset: int) -> pd.Period:
    return date.to_period("M") + int(offset)


def build_futures_ledger(staging: Path, sample: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = pd.read_csv(WRDS_METADATA, parse_dates=["lasttrddate"])
    prices = pd.read_csv(WRDS_PRICES, parse_dates=["date_"])
    if metadata["futcode"].duplicated().any() or prices.duplicated(["futcode", "date_"]).any():
        raise DiagnosticError("WRDS source keys are not unique")
    if (prices["settlement"].dropna() <= 0).any():
        raise DiagnosticError("nonpositive WRDS settlement price")
    metadata = metadata[["futcode", "dsmnem", "lasttrddate"]].copy()
    metadata["contract_month"] = metadata["lasttrddate"].dt.to_period("M")
    if metadata["contract_month"].duplicated().any():
        duplicates = metadata.loc[metadata["contract_month"].duplicated(False), "contract_month"].astype(str).tolist()
        raise DiagnosticError(f"multiple contract masters for a month: {duplicates[:10]}")
    price_groups = {
        int(futcode): frame.sort_values("date_").reset_index(drop=True)
        for futcode, frame in prices.dropna(subset=["settlement"]).groupby("futcode", sort=False)
    }
    trading_dates = sorted(
        pd.Timestamp(value) for value in prices["date_"].drop_duplicates()
        if pd.Timestamp(SAMPLE_START) <= pd.Timestamp(value) <= pd.Timestamp(SAMPLE_END)
    )
    if len(trading_dates) != 2621:
        raise DiagnosticError(f"WRDS trading-date coverage drift: {len(trading_dates)}")
    event_by_release = {
        str(row["release_date"]): str(row["generation_meeting_id"]) for row in sample["current"]
    }
    rows: list[dict[str, Any]] = []
    for trading_date in trading_dates:
        date_text = trading_date.date().isoformat()
        for convention in CONVENTIONS:
            for horizon in HORIZONS:
                if convention == "calendar_month_offset":
                    target_month = add_calendar_month(trading_date, horizon)
                    candidates = metadata.loc[metadata["contract_month"] == target_month]
                else:
                    eligible = metadata.loc[metadata["lasttrddate"] >= trading_date].sort_values(
                        ["lasttrddate", "futcode"]
                    )
                    candidates = eligible.iloc[horizon - 1:horizon]
                    target_month = candidates.iloc[0]["contract_month"] if len(candidates) == 1 else None
                base = {
                    "schema_version": SCHEMA + ":futures-return-row",
                    "convention": convention,
                    "horizon_months": int(horizon),
                    "horizon_label": f"FF{horizon}",
                    "trading_date": date_text,
                    "minutes_release_event_meeting_id": event_by_release.get(date_text),
                    "is_roster_bound_minutes_release_date": date_text in event_by_release,
                    "target_contract_month": None if target_month is None else str(target_month),
                }
                if len(candidates) != 1:
                    rows.append({**base, "status": "missing_contract", "missing_reason": f"candidate_count={len(candidates)}"})
                    continue
                contract = candidates.iloc[0]
                futcode = int(contract["futcode"])
                series = price_groups.get(futcode)
                if series is None:
                    rows.append({**base, "status": "missing_price_series", "futcode": futcode})
                    continue
                current = series.loc[series["date_"] == trading_date]
                previous = series.loc[series["date_"] < trading_date]
                if len(current) != 1:
                    rows.append({
                        **base,
                        "status": "missing_exact_release_settlement",
                        "futcode": futcode,
                        "dsmnem": str(contract["dsmnem"]),
                        "lasttrddate": contract["lasttrddate"].date().isoformat(),
                    })
                    continue
                if previous.empty:
                    rows.append({**base, "status": "missing_previous_valid_settlement", "futcode": futcode})
                    continue
                previous_row = previous.iloc[-1]
                current_price = float(current.iloc[0]["settlement"])
                previous_price = float(previous_row["settlement"])
                rows.append({
                    **base,
                    "status": "complete",
                    "missing_reason": None,
                    "futcode": futcode,
                    "dsmnem": str(contract["dsmnem"]),
                    "lasttrddate": contract["lasttrddate"].date().isoformat(),
                    "current_settlement_date": date_text,
                    "current_settlement": current_price,
                    "previous_valid_date": previous_row["date_"].date().isoformat(),
                    "previous_valid_settlement": previous_price,
                    "previous_gap_calendar_days": int((trading_date - previous_row["date_"]).days),
                    "return_log_percent": float(100.0 * math.log(current_price / previous_price)),
                    "exact_release_date": True,
                })
    if len(rows) != len(trading_dates) * len(CONVENTIONS) * len(HORIZONS):
        raise DiagnosticError("futures ledger Cartesian coverage drift")
    if len({(row["convention"], row["horizon_months"], row["trading_date"]) for row in rows}) != len(rows):
        raise DiagnosticError("duplicate futures ledger key")
    ledger_path = staging / "futures/futures_return_ledger.jsonl"
    write_new_jsonl(ledger_path, rows)
    coverage: dict[str, Any] = {}
    for convention in CONVENTIONS:
        coverage[convention] = {}
        for horizon in HORIZONS:
            subset = [row for row in rows if row["convention"] == convention and row["horizon_months"] == horizon]
            coverage[convention][f"FF{horizon}"] = {
                "expected": len(subset),
                "complete": sum(row["status"] == "complete" for row in subset),
                "status_counts": dict(sorted(Counter(str(row["status"]) for row in subset).items())),
            }
    manifest = {
        "schema_version": SCHEMA + ":futures-construction-manifest",
        "created_at_utc": utc_now(),
        "source": {
            "provider": "WRDS / LSEG Datastream Futures",
            "product": "30 DAY US FEDERAL FUNDS",
            "ticker": "FF",
            "contrcode": 331,
            "clscode": 245,
            "source_market_scope": "legacy CBOT PIT product; not a combined PIT/electronic market series",
            "manifest": file_binding(WRDS_MANIFEST),
            "contract_metadata": file_binding(WRDS_METADATA),
            "contract_prices": file_binding(WRDS_PRICES),
        },
        "sample_trading_dates": {
            "start": SAMPLE_START,
            "end": SAMPLE_END,
            "unique_wrds_dates": len(trading_dates),
        },
        "outcome": "100 * log(current-date settlement / previous-valid settlement)",
        "price_field": "settlement",
        "primary_convention": PRIMARY_CONVENTION,
        "conventions": {
            "calendar_month_offset": (
                "For FFh on each trading day, select the contract whose last-trading calendar month is "
                "h months after the current calendar month."
            ),
            "live_contract_rank": (
                "Robustness: among contracts with last-trading date on or after the current date, select the h-th "
                "contract ordered by last-trading date."
            ),
        },
        "coverage": coverage,
        "artifact": file_binding(ledger_path, relative_to=staging, rows=len(rows)),
        "exact_tadle_vendor_continuation_replication": False,
    }
    write_new_json(staging / "futures/manifest.json", manifest)
    return rows, manifest


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    if not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
        raise DiagnosticError("invalid standardization parameters")
    return mean, sd


def load_minutes_scores(backend_id: str) -> dict[str, Any]:
    path = OFFICIAL_BENCHMARK / "sentiment" / backend_id / "document_scores.jsonl"
    rows = read_jsonl(path)
    official: dict[str, float] = {}
    generated: dict[str, dict[str, dict[int, float]]] = {
        arm: defaultdict(dict) for arm in GENERATED_ARMS
    }
    for row in rows:
        arm = str(row["arm"])
        mid = str(row["meeting_id"])
        if arm == "official":
            official[mid] = float(row["score"])
        elif arm in GENERATED_ARMS:
            generated[arm][mid][int(row["replicate_id"])] = float(row["score"])
    return {"path": path, "official": official, "generated": generated}


def load_vix_daily_returns() -> dict[str, float]:
    frame = pd.read_csv(VIX_DAILY, parse_dates=["observation_date"])
    frame["VIXCLS"] = pd.to_numeric(frame["VIXCLS"], errors="coerce")
    frame = frame.dropna(subset=["VIXCLS"]).sort_values("observation_date").reset_index(drop=True)
    if (frame["VIXCLS"] <= 0).any() or frame["observation_date"].duplicated().any():
        raise DiagnosticError("VIX daily source failed positivity or uniqueness gate")
    frame["vix_log_percent_change"] = 100.0 * np.log(frame["VIXCLS"] / frame["VIXCLS"].shift(1))
    return {
        row.observation_date.date().isoformat(): float(row.vix_log_percent_change)
        for row in frame.itertuples(index=False)
        if math.isfinite(float(row.vix_log_percent_change))
    }


def build_semantic_inputs(
    staging: Path,
    sample: Mapping[str, Any],
    futures_rows: Sequence[Mapping[str, Any]],
    backend_id: str,
) -> dict[str, Any]:
    statement_path = staging / "statements/sentiment" / backend_id / "document_scores.jsonl"
    statements = {str(row["meeting_id"]): float(row["score"]) for row in read_jsonl(statement_path)}
    minutes = load_minutes_scores(backend_id)
    current_ids = tuple(str(row["generation_meeting_id"]) for row in sample["current"])
    estimation_ids = tuple(str(row["generation_meeting_id"]) for row in sample["estimation_current"])
    lag_ids = tuple(sample["prior"][mid] for mid in estimation_ids)
    needed_ids = tuple(sample["needed_ids"])
    if set(statements) != set(needed_ids):
        raise DiagnosticError(f"Statement score coverage drift: {backend_id}")
    if any(mid not in minutes["official"] for mid in needed_ids):
        raise DiagnosticError(f"official Minutes score coverage drift: {backend_id}")
    if any(
        set(minutes["generated"][arm].get(mid, {})) != set(range(5))
        for arm in GENERATED_ARMS for mid in needed_ids
    ):
        raise DiagnosticError(f"generated Minutes replicate coverage drift: {backend_id}")

    statement_scale = mean_sd([statements[mid] for mid in current_ids])
    point_minutes: dict[str, dict[str, float]] = {
        "official": {mid: minutes["official"][mid] for mid in needed_ids}
    }
    for arm in GENERATED_ARMS:
        point_minutes[arm] = {
            mid: float(np.mean([minutes["generated"][arm][mid][rep] for rep in range(5)]))
            for mid in needed_ids
        }
    minute_scales = {
        arm: mean_sd([point_minutes[arm][mid] for mid in current_ids]) for arm in ARMS
    }
    statement_z = {
        mid: (statements[mid] - statement_scale[0]) / statement_scale[1] for mid in needed_ids
    }
    point_ns: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    point_rz: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        mu, sd = minute_scales[arm]
        for mid in needed_ids:
            point_rz[arm][mid] = (point_minutes[arm][mid] - mu) / sd - statement_z[mid]
        for mid, lag in zip(estimation_ids, lag_ids, strict=True):
            point_ns[arm][mid] = point_rz[arm][mid] - TADLE_GAMMA * point_rz[arm][lag]

    vix_daily = load_vix_daily_returns()
    event_by_release = {
        str(row["release_date"]): str(row["generation_meeting_id"])
        for row in sample["estimation_current"]
    }
    lag_gap_release_dates = {
        str(row["release_date"])
        for row in sample["current"]
        if str(row["generation_meeting_id"]) in LAG_GAP_CURRENT
    }
    unavailable_event_dates = set(MISSING_MODEL_RELEASE_DATES) | lag_gap_release_dates
    future_map = {
        (str(row["convention"]), int(row["horizon_months"]), str(row["trading_date"])): row
        for row in futures_rows
    }
    panel_rows: list[dict[str, Any]] = []
    for convention in CONVENTIONS:
        for horizon in HORIZONS:
            future_subset = sorted(
                [
                    row for row in futures_rows
                    if row["convention"] == convention and row["horizon_months"] == horizon
                ],
                key=lambda row: str(row["trading_date"]),
            )
            for future in future_subset:
                if future["status"] != "complete":
                    continue
                trading_date = str(future["trading_date"])
                if trading_date not in vix_daily or trading_date in unavailable_event_dates:
                    continue
                mid = event_by_release.get(trading_date)
                for arm in ARMS:
                    panel_rows.append({
                        "schema_version": SCHEMA + ":analysis-panel-row",
                        "backend_id": backend_id,
                        "convention": convention,
                        "horizon_months": horizon,
                        "horizon_label": f"FF{horizon}",
                        "arm": arm,
                        "paper_label": PAPER_LABELS[arm],
                        "trading_date": trading_date,
                        "release_year": trading_date[:4],
                        "post_2011": int(trading_date > POST_2011_CUTOFF),
                        "is_minutes_release_event": mid is not None,
                        "event_meeting_id": mid,
                        "lag_meeting_id": None if mid is None else sample["prior"][mid],
                        "minutes_raw_score": None if mid is None else point_minutes[arm][mid],
                        "statement_raw_score": None if mid is None else statements[mid],
                        "relative_standardized_sentiment": None if mid is None else point_rz[arm][mid],
                        "news_shock": 0.0 if mid is None else point_ns[arm][mid],
                        "tadle_gamma": TADLE_GAMMA,
                        "vix_log_percent_change": vix_daily[trading_date],
                        "futures_return_log_percent": float(future["return_log_percent"]),
                        "futcode": int(future["futcode"]),
                        "dsmnem": str(future["dsmnem"]),
                    })
    scales = {
        "backend_id": backend_id,
        "sample_n": len(current_ids),
        "estimable_news_shock_events": len(estimation_ids),
        "sample_release_start": SAMPLE_START,
        "sample_release_end": SAMPLE_END,
        "statement": {"mean": statement_scale[0], "sd": statement_scale[1], "ddof": 1},
        "minutes": {
            arm: {"mean": minute_scales[arm][0], "sd": minute_scales[arm][1], "ddof": 1}
            for arm in ARMS
        },
        "standardization_contract": (
            "Tadle-form separate within-source standardization on the 83 current release events; "
            "the same parameters are applied to the one lag-buffer meeting."
        ),
    }
    return {
        "backend_id": backend_id,
        "statements": statements,
        "minutes": minutes,
        "current_ids": current_ids,
        "estimation_ids": estimation_ids,
        "lag_ids": lag_ids,
        "needed_ids": needed_ids,
        "statement_z": statement_z,
        "point_minutes": point_minutes,
        "point_rz": point_rz,
        "point_ns": point_ns,
        "minute_scales": minute_scales,
        "statement_scale": statement_scale,
        "unavailable_event_dates_excluded_from_daily_panel": sorted(unavailable_event_dates),
        "panel_rows": panel_rows,
        "scales": scales,
    }


def design_matrix(
    news: np.ndarray,
    vix: np.ndarray,
    years: Sequence[str],
    post: np.ndarray,
    *,
    interaction: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    year_levels = sorted(set(str(year) for year in years))
    columns = [np.ones(len(news), dtype=np.float64), news, vix]
    names = ["intercept", "news_shock", "vix"]
    if interaction:
        columns.extend((post * news, post))
        names.extend(("post2011_x_news_shock", "post2011"))
    for level in year_levels[1:]:
        columns.append(np.asarray([float(str(year) == level) for year in years], dtype=np.float64))
        names.append(f"year_{level}")
    matrix = np.column_stack(columns)
    return matrix, tuple(names)


def fit_ols_hc1(y: np.ndarray, matrix: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
    if int(rank) != matrix.shape[1]:
        raise DiagnosticError(f"rank-deficient point design: rank={rank}, k={matrix.shape[1]}, names={names}")
    residuals = y - matrix @ coefficients
    nobs, k = matrix.shape
    bread = np.linalg.inv(matrix.T @ matrix)
    meat = matrix.T @ ((residuals ** 2)[:, None] * matrix)
    covariance = (nobs / (nobs - k)) * bread @ meat @ bread
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    centered = y - y.mean()
    tss = float(centered @ centered)
    rss = float(residuals @ residuals)
    return {
        "coefficients": coefficients,
        "standard_errors": standard_errors,
        "covariance": covariance,
        "nobs": nobs,
        "k": k,
        "r_squared": float(1.0 - rss / tss) if tss > 0 else float("nan"),
        "names": tuple(names),
    }


def fit_basic_beta(y: np.ndarray, news: np.ndarray, vix: np.ndarray, years: Sequence[str]) -> float:
    matrix, names = design_matrix(
        news,
        vix,
        years,
        np.zeros(len(news), dtype=np.float64),
        interaction=False,
    )
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
    if int(rank) != matrix.shape[1]:
        raise DiagnosticError(f"rank-deficient basic bootstrap design: {names}")
    return float(coefficients[1])


def point_estimates(panel_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted({
        (str(row["backend_id"]), str(row["convention"]), int(row["horizon_months"]), str(row["arm"]))
        for row in panel_rows
    })
    for backend_id, convention, horizon, arm in keys:
        subset = sorted(
            [
                row for row in panel_rows
                if row["backend_id"] == backend_id
                and row["convention"] == convention
                and row["horizon_months"] == horizon
                and row["arm"] == arm
            ],
            key=lambda row: str(row["trading_date"]),
        )
        y = np.asarray([row["futures_return_log_percent"] for row in subset], dtype=np.float64)
        news = np.asarray([row["news_shock"] for row in subset], dtype=np.float64)
        vix = np.asarray([row["vix_log_percent_change"] for row in subset], dtype=np.float64)
        post = np.asarray([row["post_2011"] for row in subset], dtype=np.float64)
        years = [str(row["release_year"]) for row in subset]
        basic_matrix, basic_names = design_matrix(news, vix, years, post, interaction=False)
        basic = fit_ols_hc1(y, basic_matrix, basic_names)
        rows.append({
            "schema_version": SCHEMA + ":coefficient-row",
            "backend_id": backend_id,
            "convention": convention,
            "horizon_months": horizon,
            "horizon_label": f"FF{horizon}",
            "specification": "basic_tadle_form",
            "arm": arm,
            "paper_label": PAPER_LABELS[arm],
            "nobs": basic["nobs"],
            "beta_news_shock": float(basic["coefficients"][1]),
            "hc1_se_news_shock": float(basic["standard_errors"][1]),
            "beta_news_shock_basis_points": float(100.0 * basic["coefficients"][1]),
            "hc1_se_news_shock_basis_points": float(100.0 * basic["standard_errors"][1]),
            "beta_post2011_x_news_shock": None,
            "hc1_se_post2011_x_news_shock": None,
            "beta_news_shock_post2011": None,
            "hc1_se_news_shock_post2011": None,
            "r_squared": basic["r_squared"],
            "covariance": "HC1 heteroskedasticity-robust",
        })
        interaction_matrix, interaction_names = design_matrix(news, vix, years, post, interaction=True)
        interaction = fit_ols_hc1(y, interaction_matrix, interaction_names)
        index_news = interaction_names.index("news_shock")
        index_cross = interaction_names.index("post2011_x_news_shock")
        post_beta = float(interaction["coefficients"][index_news] + interaction["coefficients"][index_cross])
        post_var = float(
            interaction["covariance"][index_news, index_news]
            + interaction["covariance"][index_cross, index_cross]
            + 2.0 * interaction["covariance"][index_news, index_cross]
        )
        rows.append({
            "schema_version": SCHEMA + ":coefficient-row",
            "backend_id": backend_id,
            "convention": convention,
            "horizon_months": horizon,
            "horizon_label": f"FF{horizon}",
            "specification": "post2011_interaction_tadle_form",
            "arm": arm,
            "paper_label": PAPER_LABELS[arm],
            "nobs": interaction["nobs"],
            "beta_news_shock": float(interaction["coefficients"][index_news]),
            "hc1_se_news_shock": float(interaction["standard_errors"][index_news]),
            "beta_news_shock_basis_points": float(100.0 * interaction["coefficients"][index_news]),
            "hc1_se_news_shock_basis_points": float(100.0 * interaction["standard_errors"][index_news]),
            "beta_post2011_x_news_shock": float(interaction["coefficients"][index_cross]),
            "hc1_se_post2011_x_news_shock": float(interaction["standard_errors"][index_cross]),
            "beta_post2011_x_news_shock_basis_points": float(100.0 * interaction["coefficients"][index_cross]),
            "hc1_se_post2011_x_news_shock_basis_points": float(100.0 * interaction["standard_errors"][index_cross]),
            "beta_news_shock_post2011": post_beta,
            "hc1_se_news_shock_post2011": math.sqrt(max(post_var, 0.0)),
            "beta_news_shock_post2011_basis_points": 100.0 * post_beta,
            "hc1_se_news_shock_post2011_basis_points": 100.0 * math.sqrt(max(post_var, 0.0)),
            "r_squared": interaction["r_squared"],
            "covariance": "HC1 heteroskedasticity-robust",
        })
    return rows


def build_bootstrap_plan(
    sample: Mapping[str, Any],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    current_ids = tuple(str(row["generation_meeting_id"]) for row in sample["estimation_current"])
    needed_ids = tuple(sample["needed_ids"])
    years = tuple(str(year) for year in range(2004, 2016))
    rng = np.random.default_rng(seed)
    sampled_year_indexes = rng.integers(0, len(years), size=(draws, len(years)), dtype=np.int16)
    replicate_choices = rng.integers(0, 5, size=(draws, len(needed_ids), 5), dtype=np.int8)
    contract = {
        "draws": draws,
        "seed": seed,
        "generator": "numpy.random.default_rng(PCG64)",
        "calendar_year_blocks": list(years),
        "blocks_per_draw": len(years),
        "needed_meetings": list(needed_ids),
        "generated_replicates_per_meeting": 5,
        "replicate_draws_per_meeting": 5,
        "shared_across_backends_horizons_and_generated_arms": True,
        "official_and_statement_documents_resampled_within_meeting": False,
        "feature_construction_before_row_block_resampling": True,
        "standardization_parameters_fixed_at_point_sample_values": True,
    }
    plan_sha = sha256_text(
        canonical(contract)
        + sha256_bytes(sampled_year_indexes.tobytes(order="C"))
        + sha256_bytes(replicate_choices.tobytes(order="C"))
    )
    return {
        "contract": contract,
        "plan_sha256": plan_sha,
        "years": years,
        "current_ids": current_ids,
        "needed_ids": needed_ids,
        "sampled_year_indexes": sampled_year_indexes,
        "replicate_choices": replicate_choices,
    }


def bootstrap_backend(
    semantic: Mapping[str, Any],
    panel_rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    backend_id = str(semantic["backend_id"])
    current_ids = tuple(plan["current_ids"])
    needed_ids = tuple(plan["needed_ids"])
    needed_index = {mid: index for index, mid in enumerate(needed_ids)}
    current_index = np.asarray([needed_index[mid] for mid in current_ids], dtype=np.int64)
    lag_index = np.asarray([needed_index[mid] for mid in semantic["lag_ids"]], dtype=np.int64)
    statement_z = np.asarray([semantic["statement_z"][mid] for mid in needed_ids], dtype=np.float64)
    choices = np.asarray(plan["replicate_choices"])
    draws = int(plan["contract"]["draws"])
    ns_draws: dict[str, np.ndarray] = {}
    for arm in ARMS:
        mu, sd = semantic["minute_scales"][arm]
        if arm == "official":
            minute_z = np.asarray(
                [(semantic["point_minutes"][arm][mid] - mu) / sd for mid in needed_ids],
                dtype=np.float64,
            )
            rz = minute_z - statement_z
            ns = rz[current_index] - TADLE_GAMMA * rz[lag_index]
            ns_draws[arm] = np.broadcast_to(ns, (draws, len(current_ids)))
        else:
            raw = np.asarray(
                [
                    [semantic["minutes"]["generated"][arm][mid][rep] for rep in range(5)]
                    for mid in needed_ids
                ],
                dtype=np.float64,
            )
            expanded = np.broadcast_to(raw, (draws, *raw.shape))
            resampled_mean = np.take_along_axis(expanded, choices.astype(np.int64), axis=2).mean(axis=2)
            minute_z = (resampled_mean - mu) / sd
            rz = minute_z - statement_z[None, :]
            ns_draws[arm] = rz[:, current_index] - TADLE_GAMMA * rz[:, lag_index]

    panel_by_horizon: dict[int, dict[str, Any]] = {}
    for horizon in HORIZONS:
        subset = sorted(
            [
                row for row in panel_rows
                if row["backend_id"] == backend_id
                and row["convention"] == PRIMARY_CONVENTION
                and row["horizon_months"] == horizon
                and row["arm"] == "official"
            ],
            key=lambda row: str(row["trading_date"]),
        )
        event_indexes = np.asarray([
            -1 if row["event_meeting_id"] is None else current_ids.index(str(row["event_meeting_id"]))
            for row in subset
        ], dtype=np.int64)
        panel_by_horizon[horizon] = {
            "event_indexes": event_indexes,
            "y": np.asarray([row["futures_return_log_percent"] for row in subset], dtype=np.float64),
            "vix": np.asarray([row["vix_log_percent_change"] for row in subset], dtype=np.float64),
            "years": tuple(str(row["release_year"]) for row in subset),
        }

    year_values = tuple(plan["years"])
    sampled_year_indexes = np.asarray(plan["sampled_year_indexes"])
    result: list[dict[str, Any]] = []
    for draw in range(draws):
        sampled_years = tuple(year_values[int(index)] for index in sampled_year_indexes[draw])
        betas: dict[str, dict[str, float]] = {}
        for horizon in HORIZONS:
            panel = panel_by_horizon[horizon]
            selected = np.concatenate([
                np.asarray([i for i, year in enumerate(panel["years"]) if year == sampled], dtype=np.int64)
                for sampled in sampled_years
            ])
            betas[f"FF{horizon}"] = {}
            for arm in ARMS:
                news = np.zeros(len(panel["event_indexes"]), dtype=np.float64)
                event_mask = panel["event_indexes"] >= 0
                news[event_mask] = ns_draws[arm][draw, panel["event_indexes"][event_mask]]
                betas[f"FF{horizon}"][arm] = fit_basic_beta(
                    panel["y"][selected],
                    news[selected],
                    panel["vix"][selected],
                    [panel["years"][i] for i in selected],
                )
        result.append({
            "schema_version": SCHEMA + ":bootstrap-row",
            "backend_id": backend_id,
            "draw_index": draw,
            "plan_sha256": plan["plan_sha256"],
            "sampled_calendar_years": list(sampled_years),
            "specification": "basic_tadle_form",
            "convention": PRIMARY_CONVENTION,
            "betas": betas,
        })
    return result


def two_sided_p(values: np.ndarray) -> float:
    return float(beta_statistics.two_sided_bootstrap_p_value(values))


def summarize_bootstrap(
    point_rows: Sequence[Mapping[str, Any]],
    draw_rows_by_backend: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    point = {
        (str(row["backend_id"]), int(row["horizon_months"]), str(row["arm"])): float(row["beta_news_shock"])
        for row in point_rows
        if row["convention"] == PRIMARY_CONVENTION and row["specification"] == "basic_tadle_form"
    }
    contrasts: list[dict[str, Any]] = []
    gains: list[dict[str, Any]] = []
    for backend_id in BACKEND_IDS:
        draws = draw_rows_by_backend[backend_id]
        backend_contrasts: list[dict[str, Any]] = []
        backend_gains: list[dict[str, Any]] = []
        for horizon in HORIZONS:
            label = f"FF{horizon}"
            official_beta = point[(backend_id, horizon, "official")]
            for arm in GENERATED_ARMS:
                estimate = point[(backend_id, horizon, arm)] - official_beta
                values = np.asarray([
                    row["betas"][label][arm] - row["betas"][label]["official"] for row in draws
                ], dtype=np.float64)
                low, high = beta_statistics.percentile_interval(values)
                backend_contrasts.append({
                    "schema_version": SCHEMA + ":model-minus-official-row",
                    "backend_id": backend_id,
                    "family": "basic_beta_model_minus_official_across_3_models_x_4_horizons",
                    "convention": PRIMARY_CONVENTION,
                    "specification": "basic_tadle_form",
                    "horizon_months": horizon,
                    "horizon_label": label,
                    "arm": arm,
                    "paper_label": PAPER_LABELS[arm],
                    "estimate": estimate,
                    "ci_95_low": low,
                    "ci_95_high": high,
                    "estimate_basis_points": 100.0 * estimate,
                    "ci_95_low_basis_points": 100.0 * low,
                    "ci_95_high_basis_points": 100.0 * high,
                    "bootstrap_p_raw": two_sided_p(values),
                    "bootstrap_draws": len(draws),
                })
            distances = {
                arm: abs(point[(backend_id, horizon, arm)] - official_beta) for arm in GENERATED_ARMS
            }
            for comparator in ("chk1", "chk0"):
                estimate = distances[comparator] - distances["chk3"]
                values = np.asarray([
                    abs(row["betas"][label][comparator] - row["betas"][label]["official"])
                    - abs(row["betas"][label]["chk3"] - row["betas"][label]["official"])
                    for row in draws
                ], dtype=np.float64)
                low, high = beta_statistics.percentile_interval(values)
                backend_gains.append({
                    "schema_version": SCHEMA + ":distance-gain-row",
                    "backend_id": backend_id,
                    "family": "chk2_distance_gains_across_2_comparators_x_4_horizons",
                    "convention": PRIMARY_CONVENTION,
                    "specification": "basic_tadle_form",
                    "horizon_months": horizon,
                    "horizon_label": label,
                    "comparator_arm": comparator,
                    "gain_definition": (
                        f"abs(beta_{comparator}-beta_official)-abs(beta_chk3-beta_official)"
                    ),
                    "estimate": estimate,
                    "ci_95_low": low,
                    "ci_95_high": high,
                    "estimate_basis_points": 100.0 * estimate,
                    "ci_95_low_basis_points": 100.0 * low,
                    "ci_95_high_basis_points": 100.0 * high,
                    "bootstrap_p_raw": two_sided_p(values),
                    "bootstrap_draws": len(draws),
                })
        for row, adjusted in zip(
            backend_contrasts,
            beta_statistics.holm_adjust([row["bootstrap_p_raw"] for row in backend_contrasts]),
            strict=True,
        ):
            row["holm_p"] = float(adjusted)
        for row, adjusted in zip(
            backend_gains,
            beta_statistics.holm_adjust([row["bootstrap_p_raw"] for row in backend_gains]),
            strict=True,
        ):
            row["holm_p"] = float(adjusted)
        contrasts.extend(backend_contrasts)
        gains.extend(backend_gains)
    return contrasts, gains


def format_value(value: float | None, digits: int = 5) -> str:
    return "--" if value is None else f"{value:+.{digits}f}"


def render_tables(
    point_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    gains: Sequence[Mapping[str, Any]],
    futures_manifest: Mapping[str, Any],
) -> str:
    lines = [
        "# Tadle-Form Federal Funds Futures Diagnostic: Result Tables",
        "",
        "The primary tables use the calendar-month-offset contract convention and the basic Tadle-form equation.",
        "Coefficients are historical document-source diagnostics; generated documents were not released.",
        "",
    ]
    for backend_id in BACKEND_IDS:
        lines.extend([
            f"## {backend_id}: basic news-shock coefficients",
            "",
            "| Horizon | N | Official Minutes (bp) | Model chk-0 (bp) | Model chk-1 cp200 (bp) | Model chk-2 cp318 (bp) |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for horizon in HORIZONS:
            subset = {
                str(row["arm"]): row for row in point_rows
                if row["backend_id"] == backend_id
                and row["convention"] == PRIMARY_CONVENTION
                and row["specification"] == "basic_tadle_form"
                and row["horizon_months"] == horizon
            }
            cells = []
            for arm in ARMS:
                row = subset[arm]
                cells.append(
                    f"{format_value(row['beta_news_shock_basis_points'], 4)} "
                    f"({row['hc1_se_news_shock_basis_points']:.4f})"
                )
            lines.append(f"| FF{horizon} | {subset['official']['nobs']} | " + " | ".join(cells) + " |")
        lines.extend([
            "",
            "Coefficients are reported in basis points per one-unit news shock; parentheses contain HC1 heteroskedasticity-robust standard errors in basis points.",
            "",
            "### Model-minus-official paired contrasts",
            "",
            "| Horizon | Model | Difference | 95% paired year-block bootstrap CI | Raw p | Holm p |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for row in contrasts:
            if row["backend_id"] != backend_id:
                continue
            lines.append(
                f"| {row['horizon_label']} | {row['paper_label']} | {format_value(row['estimate_basis_points'], 4)} | "
                f"[{format_value(row['ci_95_low_basis_points'], 4)}, {format_value(row['ci_95_high_basis_points'], 4)}] | "
                f"{row['bootstrap_p_raw']:.6f} | {row['holm_p']:.6f} |"
            )
        lines.extend([
            "",
            "### Model chk-2 absolute-distance gains",
            "",
            "| Horizon | Comparator | Gain | 95% paired year-block bootstrap CI | Raw p | Holm p |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for row in gains:
            if row["backend_id"] != backend_id:
                continue
            comparator = PAPER_LABELS[str(row["comparator_arm"])]
            lines.append(
                f"| {row['horizon_label']} | {comparator} | {format_value(row['estimate_basis_points'], 4)} | "
                f"[{format_value(row['ci_95_low_basis_points'], 4)}, {format_value(row['ci_95_high_basis_points'], 4)}] | "
                f"{row['bootstrap_p_raw']:.6f} | {row['holm_p']:.6f} |"
            )
        lines.append("")
    lines.extend([
        "## Futures coverage",
        "",
        "| Convention | Horizon | Complete daily returns | Expected WRDS trading dates |",
        "|---|---|---:|---:|",
    ])
    for convention in CONVENTIONS:
        for horizon in HORIZONS:
            coverage = futures_manifest["coverage"][convention][f"FF{horizon}"]
            lines.append(f"| {convention} | FF{horizon} | {coverage['complete']} | {coverage['expected']} |")
    return "\n".join(lines) + "\n"


def result_summary(
    point_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    gains: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    primary_contrasts = [row for row in contrasts if row["backend_id"] == official_runner.DISTIL_ID]
    primary_gains = [row for row in gains if row["backend_id"] == official_runner.DISTIL_ID]
    return {
        "primary_backend": official_runner.DISTIL_ID,
        "primary_convention": PRIMARY_CONVENTION,
        "primary_model_minus_official_holm_significant_count": sum(row["holm_p"] < 0.05 for row in primary_contrasts),
        "primary_model_minus_official_comparison_count": len(primary_contrasts),
        "primary_chk2_distance_gain_holm_significant_count": sum(row["holm_p"] < 0.05 for row in primary_gains),
        "primary_chk2_distance_gain_comparison_count": len(primary_gains),
        "smallest_primary_model_minus_official_holm_p": min(row["holm_p"] for row in primary_contrasts),
        "smallest_primary_distance_gain_holm_p": min(row["holm_p"] for row in primary_gains),
        "point_estimate_rows": len(point_rows),
    }


def render_technical_report(
    summary: Mapping[str, Any],
    futures_manifest: Mapping[str, Any],
    statement_rows: Sequence[Mapping[str, Any]],
    point_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    gains: Sequence[Mapping[str, Any]],
    draws: int,
) -> str:
    primary_sig = int(summary["primary_model_minus_official_holm_significant_count"])
    gain_sig = int(summary["primary_chk2_distance_gain_holm_significant_count"])
    primary_n = {
        int(row["horizon_months"]): int(row["nobs"])
        for row in point_rows
        if row["backend_id"] == official_runner.DISTIL_ID
        and row["convention"] == PRIMARY_CONVENTION
        and row["specification"] == "basic_tadle_form"
        and row["arm"] == "official"
    }
    return f"""# Tadle-Form Federal Funds Futures Document-Source Diagnostic

## Technical summary

This analysis completed the Tadle-form Federal Funds futures test on the full available daily trading panel from December 1, 2004 through April 30, 2015. The sentiment event schedule contains 83 roster-bound Minutes releases, of which 82 have a valid immediately preceding meeting in the frozen model-output roster. The primary DistilBERT FOMC stance specification found {primary_sig} Holm-significant model-minus-official coefficient differences among 12 comparisons and {gain_sig} Holm-significant Model chk-2 coefficient-distance gains among 8 comparisons. A non-significant difference is not evidence of equality or equivalence.

The exercise is an adaptation, not an exact replication of Tadle (2022). It implements the documented relative-sentiment and news-shock equations, uses official Statements and full official Minutes, and uses 1-, 3-, 6-, and 12-month Federal Funds futures constructed from WRDS Datastream settlements. It does not reproduce Tadle's original sentiment dictionary, proprietary vendor continuation identifiers, or published data vintage. Generated Minutes were never released and therefore could not have caused historical futures returns.

## Data coverage and document construction

The document sample contains 83 current meetings plus one pre-window lag-buffer meeting. All {len(statement_rows)} official Statements were downloaded from Federal Reserve URLs, sealed as raw HTML, cleaned to the substantive release body, and scored without truncation. Official Minutes and the three generated sources reuse the sealed full-document scores from the 256-meeting benchmark; generated meeting documents average five frozen stochastic replicates for point estimation.

For the primary calendar-month-offset convention, the final regression counts are FF1 N={primary_n[1]}, FF3 N={primary_n[3]}, FF6 N={primary_n[6]}, and FF12 N={primary_n[12]}. FF12 is materially shorter because a twelve-month-ahead raw contract is not available for many early trading dates. Six additional dates without a usable VIX change and two dates affected by the missing 2009-11-03/04 model meeting are excluded uniformly where otherwise present. Missing outcomes and unavailable event shocks are disclosed and never imputed.

## Tadle-form sentiment and market equations

For document source a, Minutes and Statement scores are standardized separately over the 83 current release events. Relative sentiment is RZ_a,t = Z^M_a,t - Z^S_t, and the frozen news shock is NS_a,t = RZ_a,t - 0.368 RZ_a,t-1. On trading days without a roster-bound Minutes release, NS is set to zero; this event-day coding is inferred from Tadle's all-trading-day sample sizes rather than explicitly documented in the precursor text. The basic market equation regresses the daily log-percent Federal Funds futures settlement return on the news shock, the daily VIX log-percent change, and calendar-year fixed effects. The interaction specification adds a post-August-8-2011 indicator and its interaction with the news shock. Point-estimate covariance is HC1 heteroskedasticity robust.

The primary outcome convention selects, on each trading date, the contract whose last-trading calendar month is h months after the current month. A separately reported live-contract-rank convention is a robustness construction. Neither convention is claimed to reproduce Tadle's exact proprietary continuous-series roll rule.

## Paired coefficient inference

The project-specific comparison layer uses {draws:,} paired calendar-year block-bootstrap draws for the basic equation. The same sampled years and meeting-level replicate draws are shared across all document sources, backends, and horizons. Official Minutes and Statements are deterministic within meeting. Standardization parameters and chronological lag identities are frozen before resampling. Coefficients and contrasts are reported in basis points, obtained by multiplying log-percent coefficients by 100. Model-minus-official p-values are Holm adjusted across three generated sources and four horizons within each backend; Model chk-2 distance gains form a separate eight-test family.

The post-2011 interaction model is reported with HC1 standard errors but is not included in the year-block bootstrap. Resampling complete years can omit 2011, the only year containing observations on both sides of the August 8 cutoff, which makes the post indicator unidentified conditional on year fixed effects in many bootstrap draws.

## Interpretation boundaries

The coefficients describe associations between a historical return series and sentiment measures recovered from alternative document sources. The official Minutes are historical releases; the generated sources are counterfactual documents assembled from eight Core8 paragraphs and contain a narrower information scope. Accordingly, coefficient agreement does not establish semantic equivalence, information sufficiency, market impact, or causality. Confidence intervals that include zero show that the frozen design did not distinguish a coefficient difference; they do not prove equivalence.

## Reproducibility and remaining questions

All source pages, hashes, Statement scores, daily futures selections, analysis rows, coefficient estimates, bootstrap draws, and multiplicity-adjusted contrasts are sealed under this release. The validation report replays coverage, key uniqueness, formulas, and result-table identities. A literal replication would still require the exact published article's sentiment dictionary, removal of the repeated Statement passage from each official Minutes document, the original data vintage, and vendor-specific FF1/FF3/FF6/FF12 continuation rules.
"""


def validate_staging(
    staging: Path,
    sample: Mapping[str, Any],
    statement_rows: Sequence[Mapping[str, Any]],
    futures_rows: Sequence[Mapping[str, Any]],
    panel_rows: Sequence[Mapping[str, Any]],
    point_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    gains: Sequence[Mapping[str, Any]],
    draw_rows_by_backend: Mapping[str, Sequence[Mapping[str, Any]]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "sample_current_meetings_83": len(sample["current"]) == 83,
        "estimable_news_shock_events_82": len(sample["estimation_current"]) == 82,
        "sample_current_plus_lag_meetings_84": len(sample["needed_ids"]) == 84,
        "statement_sources_unique_complete": len(statement_rows) == 84 and len({row["meeting_id"] for row in statement_rows}) == 84,
        "statement_bodies_nonempty": all(int(row["characters"]) >= 200 for row in statement_rows),
        "statement_chrome_absent": all(
            marker.casefold() not in document["document_text"].casefold()
            for document in read_jsonl(staging / "statements/documents.jsonl")
            for marker in ("Last Update:", "Home | News and events", "javascript is disabled")
        ),
        "statement_raw_hashes_match": all(
            sha256_file(staging / str(row["raw_path"])) == row["raw_sha256"] for row in statement_rows
        ),
        "futures_cartesian_keys_complete": len(futures_rows) == 2621 * 2 * 4,
        "futures_keys_unique": len({
            (row["convention"], row["horizon_months"], row["trading_date"]) for row in futures_rows
        }) == len(futures_rows),
        "complete_futures_returns_finite": all(
            math.isfinite(float(row["return_log_percent"])) for row in futures_rows if row["status"] == "complete"
        ),
        "panel_keys_unique": len({
            (row["backend_id"], row["convention"], row["horizon_months"], row["arm"], row["trading_date"])
            for row in panel_rows
        }) == len(panel_rows),
        "point_rows_expected": len(point_rows) == 2 * 2 * 4 * 4 * 2,
        "contrast_rows_expected": len(contrasts) == 2 * 4 * 3,
        "distance_gain_rows_expected": len(gains) == 2 * 4 * 2,
        "bootstrap_draws_complete": all(
            len(draw_rows_by_backend[backend_id]) == int(plan["contract"]["draws"])
            for backend_id in BACKEND_IDS
        ),
        "bootstrap_plan_bound": all(
            row["plan_sha256"] == plan["plan_sha256"]
            for backend_id in BACKEND_IDS for row in draw_rows_by_backend[backend_id]
        ),
        "holm_probabilities_valid": all(0.0 <= float(row["holm_p"]) <= 1.0 for row in [*contrasts, *gains]),
    }
    status = "passed" if all(checks.values()) else "failed"
    report = {
        "schema_version": SCHEMA + ":validation-report",
        "created_at_utc": utc_now(),
        "status": status,
        "checks": checks,
        "limitations": [
            "This is a Tadle-form adaptation, not a literal replication of the 2022 publication.",
            "The primary futures horizon mapping is an explicit calendar-month convention, not a verified proprietary continuation rule.",
            "The WRDS source is the legacy CBOT PIT product and does not combine the later electronic ZQ product.",
            "Generated documents were not historically released; no generated-text coefficient is a market impact.",
            "Year-block bootstrap inference is limited to the basic equation; the post-2011 interaction uses HC1 standard errors.",
            "The project retains full official Minutes bodies, whereas Tadle removed the passage repeating the earlier Statement.",
        ],
    }
    if status != "passed":
        raise DiagnosticError(f"staging validation failed: {checks}")
    return report


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"create-only output root already exists: {args.output_root}")
    if args.bootstrap_draws <= 0 or args.batch_size <= 0:
        raise ValueError("bootstrap draws and batch size must be positive")
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_root.name}.staging.", dir=args.output_root.parent))
    completed = False
    try:
        runtime = runtime_contract()
        sample = load_sample()
        statement_rows, statement_documents = fetch_statements(staging, sample)
        statement_manifests = {
            backend_id: score_statement_backend(
                staging,
                statement_documents,
                backend_id,
                device=args.device,
                batch_size=args.batch_size,
            )
            for backend_id in BACKEND_IDS
        }
        futures_rows, futures_manifest = build_futures_ledger(staging, sample)

        semantics = {
            backend_id: build_semantic_inputs(staging, sample, futures_rows, backend_id)
            for backend_id in BACKEND_IDS
        }
        panel_rows = [
            row for backend_id in BACKEND_IDS for row in semantics[backend_id]["panel_rows"]
        ]
        write_new_jsonl(staging / "estimation/analysis_panel.jsonl", panel_rows)
        write_new_json(
            staging / "estimation/standardization_scales.json",
            {backend_id: semantics[backend_id]["scales"] for backend_id in BACKEND_IDS},
        )

        point_rows = point_estimates(panel_rows)
        write_new_jsonl(staging / "estimation/coefficient_estimates.jsonl", point_rows)
        plan = build_bootstrap_plan(sample, draws=args.bootstrap_draws, seed=args.bootstrap_seed)
        plan_public = {**plan["contract"], "plan_sha256": plan["plan_sha256"]}
        write_new_json(staging / "estimation/bootstrap_plan.json", plan_public)
        draw_rows_by_backend: dict[str, list[dict[str, Any]]] = {}
        for backend_id in BACKEND_IDS:
            draw_rows = bootstrap_backend(semantics[backend_id], panel_rows, plan)
            draw_rows_by_backend[backend_id] = draw_rows
            write_new_jsonl(staging / f"estimation/{backend_id}/bootstrap_draws.jsonl", draw_rows)
        contrasts, gains = summarize_bootstrap(point_rows, draw_rows_by_backend)
        write_new_jsonl(staging / "estimation/model_minus_official_contrasts.jsonl", contrasts)
        write_new_jsonl(staging / "estimation/distance_gains.jsonl", gains)
        summary = result_summary(point_rows, contrasts, gains)
        write_new_json(staging / "result_summary.json", summary)
        write_new_text(
            staging / "result_tables.md",
            render_tables(point_rows, contrasts, gains, futures_manifest),
        )
        write_new_text(
            staging / "technical_report.md",
            render_technical_report(
                summary,
                futures_manifest,
                statement_rows,
                point_rows,
                contrasts,
                gains,
                args.bootstrap_draws,
            ),
        )
        validation = validate_staging(
            staging,
            sample,
            statement_rows,
            futures_rows,
            panel_rows,
            point_rows,
            contrasts,
            gains,
            draw_rows_by_backend,
            plan,
        )
        write_new_json(staging / "validation_report.json", validation)
        manifest = {
            "schema_version": SCHEMA + ":release-manifest",
            "created_at_utc": utc_now(),
            "status": "complete",
            "interpretation": "historical reduced-form document-source coefficient diagnostic",
            "exact_tadle_2022_replication": False,
            "sample": {
                "release_start": SAMPLE_START,
                "release_end": SAMPLE_END,
                "wrds_trading_dates": 2621,
                "current_meetings": len(sample["current"]),
                "estimable_news_shock_events": len(sample["estimation_current"]),
                "lag_buffer_meetings": 1,
                "current_plus_lag_unique_meetings": len(sample["needed_ids"]),
                "lag_gap_exclusions": sample["lag_gap_exclusions"],
                "missing_model_release_dates_excluded": MISSING_MODEL_RELEASE_DATES,
            },
            "formula": {
                "relative_sentiment": "RZ[a,t] = Z_minutes[a,t] - Z_statement[t]",
                "news_shock": f"NS[a,t] = RZ[a,t] - {TADLE_GAMMA} * RZ[a,t-1]",
                "basic": "r[f,t] = alpha + beta_NS NS[a,t] + beta_VIX VIX[t] + year fixed effects + error",
                "interaction": (
                    "r[f,t] = alpha + beta_NS NS[a,t] + beta_postxNS post2011[t]*NS[a,t] "
                    "+ beta_VIX VIX[t] + beta_post post2011[t] + year fixed effects + error"
                ),
            },
            "primary_backend": official_runner.DISTIL_ID,
            "robustness_backend": official_runner.FINBERT_ID,
            "primary_futures_convention": PRIMARY_CONVENTION,
            "runtime": runtime,
            "input_bindings": {
                "implementation": file_binding(IMPLEMENTATION),
                "release_ledger": file_binding(RELEASE_LEDGER),
                "market_panel": file_binding(MARKET_PANEL),
                "vix_daily": file_binding(VIX_DAILY),
                "wrds_manifest": file_binding(WRDS_MANIFEST),
                "wrds_contract_metadata": file_binding(WRDS_METADATA),
                "wrds_contract_prices": file_binding(WRDS_PRICES),
                "official_minutes_scores": {
                    backend_id: file_binding(
                        OFFICIAL_BENCHMARK / "sentiment" / backend_id / "document_scores.jsonl"
                    ) for backend_id in BACKEND_IDS
                },
            },
            "statement_sentiment_manifests": statement_manifests,
            "bootstrap": plan_public,
            "result_summary": summary,
            "artifacts": {
                "statement_source_manifest": file_binding(
                    staging / "statements/source_manifest.jsonl", relative_to=staging, rows=len(statement_rows)
                ),
                "statement_documents": file_binding(
                    staging / "statements/documents.jsonl", relative_to=staging, rows=len(statement_documents)
                ),
                "futures_return_ledger": file_binding(
                    staging / "futures/futures_return_ledger.jsonl", relative_to=staging, rows=len(futures_rows)
                ),
                "analysis_panel": file_binding(
                    staging / "estimation/analysis_panel.jsonl", relative_to=staging, rows=len(panel_rows)
                ),
                "standardization_scales": file_binding(
                    staging / "estimation/standardization_scales.json", relative_to=staging
                ),
                "coefficient_estimates": file_binding(
                    staging / "estimation/coefficient_estimates.jsonl", relative_to=staging, rows=len(point_rows)
                ),
                "bootstrap_plan": file_binding(
                    staging / "estimation/bootstrap_plan.json", relative_to=staging
                ),
                "bootstrap_draws": {
                    backend_id: file_binding(
                        staging / f"estimation/{backend_id}/bootstrap_draws.jsonl",
                        relative_to=staging,
                        rows=len(draw_rows_by_backend[backend_id]),
                    ) for backend_id in BACKEND_IDS
                },
                "contrasts": file_binding(
                    staging / "estimation/model_minus_official_contrasts.jsonl",
                    relative_to=staging,
                    rows=len(contrasts),
                ),
                "distance_gains": file_binding(
                    staging / "estimation/distance_gains.jsonl", relative_to=staging, rows=len(gains)
                ),
                "result_tables": file_binding(staging / "result_tables.md", relative_to=staging),
                "technical_report": file_binding(staging / "technical_report.md", relative_to=staging),
                "validation_report": file_binding(staging / "validation_report.json", relative_to=staging),
            },
            "limitations": validation["limitations"],
        }
        write_new_json(staging / "manifest.json", manifest)
        staging.rename(args.output_root)
        completed = True
        print(canonical({
            "status": "complete",
            "output_root": str(args.output_root.resolve()),
            "summary": summary,
        }))
        return 0
    finally:
        if not completed and staging.exists():
            print(f"Partial staging retained for audit: {staging}", file=os.sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
