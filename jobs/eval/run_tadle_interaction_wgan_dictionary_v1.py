#!/usr/bin/env python3
"""Run a create-only Tadle-interaction diagnostic with the WGAN LM dictionary.

The regression equation follows the Federal Funds futures interaction
specification in Tadle's author-primary dissertation precursor.  Text is
scored with the historical WGAN dictionary contract and the current, hash-
bound Loughran--McDonald dictionary.  This is not a literal replication of
Tadle (2022): the dictionary, document scope, futures continuation series,
and bootstrap are project-specific.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from jobs.eval import chk3_beta_statistics as beta_statistics
from jobs.eval import run_tadle_form_ff_futures_v1 as base


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = Path(__file__).resolve()
SOURCE_RELEASE = (
    ROOT / "output/evaluation/main/tadle_form_official_generated_ff_futures_2004_2015_v1"
)
DOCUMENT_RELEASE = (
    ROOT / "output/evaluation/main/official_minutes_full_document_treasury_beta_1993_2025_v1"
)
WGAN_REPOSITORY = ROOT.parent / "wgan_option-rq123-london-timezone"
DICTIONARY = WGAN_REPOSITORY / "data/reference/Loughran-McDonald_MasterDictionary_1993-2025.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "output/evaluation/main/"
    "tadle_interaction_wgan_dictionary_ff_futures_2004_2015_v1"
)
SOURCE_PANEL = SOURCE_RELEASE / "estimation/analysis_panel.jsonl"
STATEMENT_DOCUMENTS = SOURCE_RELEASE / "statements/documents.jsonl"
OFFICIAL_DOCUMENTS = DOCUMENT_RELEASE / "documents/official_documents.jsonl"
GENERATED_DOCUMENTS = DOCUMENT_RELEASE / "documents/generated_documents.jsonl"
SOURCE_MANIFEST = SOURCE_RELEASE / "manifest.json"
HORIZONS = (1, 3, 6, 12)
ARMS = ("official", "chk0", "chk1", "chk3")
GENERATED_ARMS = ("chk0", "chk1", "chk3")
LABELS = {
    "official": "Official FOMC Minutes",
    "chk0": "Model chk-0",
    "chk1": "Model chk-1 cp200",
    "chk3": "Model chk-2 cp318",
}
BACKEND_ID = "wgan_historical_lm_polarity_v1"
CONVENTION = "calendar_month_offset"
SPECIFICATION = "tadle_post2011_interaction"
SCHEMA = "tadle-interaction-wgan-dictionary-ff-futures-v1"
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]*")
TADLE_GAMMA = 0.368
POST_CUTOFF = "2011-08-08"
WGAN_SCORER_COMMIT = "2619e5e"


class AnalysisError(RuntimeError):
    """A frozen analytical contract failed closed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_824)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise AnalysisError(f"non-object JSONL row: {path}")
    return rows


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(dict(row)) + "\n")
            count += 1
    return count


def binding(path: Path, *, relative_to: Path | None = None, rows: int | None = None) -> dict[str, Any]:
    resolved = path.resolve()
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


def active(value: str | None) -> bool:
    text = "" if value is None else str(value).strip()
    if not text:
        return False
    try:
        return float(text) > 0.0
    except ValueError:
        return text.lower() not in {"false", "no", "none", "nan", "0"}


def load_dictionary() -> tuple[set[str], set[str], dict[str, Any]]:
    if not DICTIONARY.is_file():
        raise FileNotFoundError(DICTIONARY)
    positive: set[str] = set()
    negative: set[str] = set()
    with DICTIONARY.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Word", "Positive", "Negative"}
        if not required.issubset(reader.fieldnames or []):
            raise AnalysisError(f"dictionary schema drift: {reader.fieldnames}")
        row_count = 0
        for row in reader:
            row_count += 1
            word = str(row["Word"]).strip().lower()
            if word and active(row["Positive"]):
                positive.add(word)
            if word and active(row["Negative"]):
                negative.add(word)
    if row_count != 86_553 or len(positive) != 347 or len(negative) != 2_345:
        raise AnalysisError(
            f"dictionary inventory drift: rows={row_count}, positive={len(positive)}, negative={len(negative)}"
        )
    if positive & negative:
        raise AnalysisError("positive and negative lexemes overlap")
    return positive, negative, {
        "dictionary_rows": row_count,
        "positive_unique_lexemes": len(positive),
        "negative_unique_lexemes": len(negative),
        "sha256": sha256_file(DICTIONARY),
    }


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(str(text))]


def score_text(text: str, positive: set[str], negative: set[str]) -> dict[str, Any]:
    tokens = tokenize(text)
    p = sum(token in positive for token in tokens)
    n = sum(token in negative for token in tokens)
    denominator = max(1, p + n)
    return {
        "token_count": len(tokens),
        "positive_count": p,
        "negative_count": n,
        "dictionary_count": p + n,
        "polarity": float((p - n) / denominator),
    }


def candidate_statement_spans(minutes: str) -> list[tuple[int, int]]:
    lowered = minutes.lower()
    marker = "the vote encompassed approval"
    starts: list[int] = []
    cursor = 0
    while True:
        found = lowered.find(marker, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + len(marker)
    spans: list[tuple[int, int]] = []
    end_markers = (
        "votes for this action:",
        "voting for this action:",
        "votes for this action.",
        "voting for this action.",
    )
    for start in starts:
        ends = [lowered.find(marker_text, start) for marker_text in end_markers]
        ends = [value for value in ends if value >= 0]
        if ends:
            spans.append((start, min(ends)))
    return spans


def remove_repeated_statement(minutes: str, statement: str) -> tuple[str, dict[str, Any]]:
    spans = candidate_statement_spans(minutes)
    if not spans:
        raise AnalysisError("no repeated-Statement candidate span")
    statement_tokens = tokenize(statement)
    candidates: list[tuple[float, int, int, int]] = []
    for index, (start, end) in enumerate(spans):
        candidate_tokens = tokenize(minutes[start:end])
        similarity = SequenceMatcher(None, statement_tokens, candidate_tokens, autojunk=False).ratio()
        candidates.append((float(similarity), index, start, end))
    candidates.sort(reverse=True)
    similarity, selected_index, start, end = candidates[0]
    if similarity <= 0.10:
        raise AnalysisError(f"repeated-Statement match too weak: {similarity}")
    if len(candidates) > 1 and math.isclose(similarity, candidates[1][0], abs_tol=1e-12):
        raise AnalysisError("ambiguous repeated-Statement match")
    cleaned = re.sub(r"\s+", " ", (minutes[:start] + " " + minutes[end:])).strip()
    return cleaned, {
        "candidate_count": len(spans),
        "selected_candidate_index": selected_index,
        "similarity_ratio": similarity,
        "runner_up_similarity_ratio": None if len(candidates) == 1 else candidates[1][0],
        "removed_start": start,
        "removed_end": end,
        "removed_characters": end - start,
        "original_text_sha256": sha256_bytes(minutes.encode("utf-8")),
        "cleaned_text_sha256": sha256_bytes(cleaned.encode("utf-8")),
    }


def load_source_skeleton() -> dict[str, Any]:
    all_rows = read_jsonl(SOURCE_PANEL)
    source_backend = base.official_runner.DISTIL_ID
    skeleton = [
        row for row in all_rows
        if row["backend_id"] == source_backend
        and row["convention"] == CONVENTION
        and row["arm"] == "official"
    ]
    by_horizon = {
        horizon: sorted(
            [row for row in skeleton if int(row["horizon_months"]) == horizon],
            key=lambda row: str(row["trading_date"]),
        )
        for horizon in HORIZONS
    }
    expected_n = {1: 2613, 3: 2612, 6: 2612, 12: 2338}
    if {h: len(rows) for h, rows in by_horizon.items()} != expected_n:
        raise AnalysisError("source panel horizon inventory drift")
    events = [row for row in by_horizon[1] if row["event_meeting_id"] is not None]
    current_ids = tuple(str(row["event_meeting_id"]) for row in events)
    lag_ids = tuple(str(row["lag_meeting_id"]) for row in events)
    if len(current_ids) != 82 or len(set(current_ids)) != 82:
        raise AnalysisError("estimable event inventory drift")
    needed_ids = tuple(sorted(set(current_ids) | set(lag_ids)))
    if len(needed_ids) != 84:
        raise AnalysisError("current/lag meeting inventory drift")
    lag_map = dict(zip(current_ids, lag_ids, strict=True))
    return {
        "by_horizon": by_horizon,
        "current_ids": current_ids,
        "lag_ids": lag_ids,
        "needed_ids": needed_ids,
        "lag_map": lag_map,
    }


def build_scores(staging: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    positive, negative, dictionary_manifest = load_dictionary()
    statement_rows = read_jsonl(STATEMENT_DOCUMENTS)
    official_rows = read_jsonl(OFFICIAL_DOCUMENTS)
    generated_rows = read_jsonl(GENERATED_DOCUMENTS)
    needed = set(source["needed_ids"])
    statements = {str(row["meeting_end_date"]): row for row in statement_rows if row["meeting_end_date"] in needed}
    official = {str(row["meeting_end_date"]): row for row in official_rows if row["meeting_end_date"] in needed}
    if set(statements) != needed or set(official) != needed:
        raise AnalysisError("official/Statement meeting join is incomplete")

    score_rows: list[dict[str, Any]] = []
    removal_rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {"statement": {}, "official": {}, "generated": {arm: {} for arm in GENERATED_ARMS}}
    for meeting_id in source["needed_ids"]:
        statement_text = str(statements[meeting_id]["document_text"])
        statement_score = score_text(statement_text, positive, negative)
        raw["statement"][meeting_id] = statement_score["polarity"]
        score_rows.append({
            "schema_version": SCHEMA + ":document-score-row",
            "document_source": "official_statement",
            "arm": "statement",
            "meeting_id": meeting_id,
            "replicate_id": None,
            **statement_score,
        })
        cleaned, removal = remove_repeated_statement(
            str(official[meeting_id]["document_text"]), statement_text
        )
        official_score = score_text(cleaned, positive, negative)
        raw["official"][meeting_id] = official_score["polarity"]
        score_rows.append({
            "schema_version": SCHEMA + ":document-score-row",
            "document_source": "official_minutes_statement_removed",
            "arm": "official",
            "meeting_id": meeting_id,
            "replicate_id": None,
            **official_score,
        })
        removal_rows.append({
            "schema_version": SCHEMA + ":statement-removal-row",
            "meeting_id": meeting_id,
            **removal,
            "cleaned_token_count": official_score["token_count"],
        })

    inventory: dict[tuple[str, str], dict[int, tuple[int, float]]] = defaultdict(dict)
    for row in generated_rows:
        arm = str(row["arm"])
        meeting_id = str(row["meeting_end_date"])
        if arm not in GENERATED_ARMS or meeting_id not in needed:
            continue
        replicate_id = int(row["replicate_id"])
        if replicate_id in inventory[(arm, meeting_id)]:
            raise AnalysisError(f"duplicate generated replicate: {arm}/{meeting_id}/{replicate_id}")
        values = score_text(str(row["document_text"]), positive, negative)
        inventory[(arm, meeting_id)][replicate_id] = (int(row["replicate_seed"]), values["polarity"])
        score_rows.append({
            "schema_version": SCHEMA + ":document-score-row",
            "document_source": "generated_core8_document",
            "arm": arm,
            "paper_label": LABELS[arm],
            "meeting_id": meeting_id,
            "replicate_id": replicate_id,
            "replicate_seed": int(row["replicate_seed"]),
            **values,
        })
    for meeting_id in source["needed_ids"]:
        expected = set(range(5))
        seeds: dict[int, int] = {}
        for arm in GENERATED_ARMS:
            observed = inventory[(arm, meeting_id)]
            if set(observed) != expected:
                raise AnalysisError(f"K=5 inventory drift: {arm}/{meeting_id}/{sorted(observed)}")
            raw["generated"][arm][meeting_id] = {
                rep: observed[rep][1] for rep in sorted(observed)
            }
            for rep, (seed, _) in observed.items():
                if rep in seeds and seeds[rep] != seed:
                    raise AnalysisError(f"replicate seed pairing drift: {meeting_id}/{rep}")
                seeds[rep] = seed

    write_jsonl(staging / "sentiment/document_scores.jsonl", score_rows)
    write_jsonl(staging / "sentiment/official_statement_removal_ledger.jsonl", removal_rows)
    write_json(staging / "sentiment/dictionary_manifest.json", {
        "schema_version": SCHEMA + ":dictionary-manifest",
        "backend_id": BACKEND_ID,
        "dictionary": dictionary_manifest,
        "token_regex": TOKEN_PATTERN.pattern,
        "case": "lowercase exact lexical match",
        "score": "(positive_count-negative_count)/max(1,positive_count+negative_count)",
        "fallback_allowed": False,
        "historical_wgan_scorer_commit": WGAN_SCORER_COMMIT,
        "provenance_note": (
            "The scorer contract is recovered from historical WGAN code; the current dictionary CSV "
            "is separately SHA-256 bound and was not contained in that commit."
        ),
    })
    if len(score_rows) != 1_428 or len(removal_rows) != 84:
        raise AnalysisError("score ledger row-count drift")
    return {
        "raw": raw,
        "score_rows": score_rows,
        "removal_rows": removal_rows,
        "dictionary": dictionary_manifest,
    }


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or len(array) < 2:
        raise AnalysisError("invalid standardization vector")
    sd = float(array.std(ddof=1))
    if sd <= 0:
        raise AnalysisError("zero standardization scale")
    return float(array.mean()), sd


def construct_semantics(source: Mapping[str, Any], scores: Mapping[str, Any]) -> dict[str, Any]:
    raw = scores["raw"]
    needed_ids = tuple(source["needed_ids"])
    scale_ids = tuple(sorted(raw["statement"])[1:])
    if len(scale_ids) != 83 or set(source["current_ids"]) - set(scale_ids):
        raise AnalysisError("83-meeting standardization population drift")
    point_minutes: dict[str, dict[str, float]] = {
        "official": dict(raw["official"]),
    }
    for arm in GENERATED_ARMS:
        point_minutes[arm] = {
            meeting_id: float(np.mean([raw["generated"][arm][meeting_id][rep] for rep in range(5)]))
            for meeting_id in needed_ids
        }
    statement_mean, statement_sd = mean_sd([raw["statement"][mid] for mid in scale_ids])
    minute_scales = {
        arm: mean_sd([point_minutes[arm][mid] for mid in scale_ids]) for arm in ARMS
    }
    statement_z = {
        mid: (raw["statement"][mid] - statement_mean) / statement_sd for mid in needed_ids
    }
    rz: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    ns: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        mean, sd = minute_scales[arm]
        for mid in needed_ids:
            rz[arm][mid] = (point_minutes[arm][mid] - mean) / sd - statement_z[mid]
        for mid in source["current_ids"]:
            ns[arm][mid] = rz[arm][mid] - TADLE_GAMMA * rz[arm][source["lag_map"][mid]]
    return {
        "scale_ids": scale_ids,
        "point_minutes": point_minutes,
        "statement_z": statement_z,
        "statement_scale": (statement_mean, statement_sd),
        "minute_scales": minute_scales,
        "rz": rz,
        "ns": ns,
    }


def build_panel(source: Mapping[str, Any], semantics: Mapping[str, Any]) -> list[dict[str, Any]]:
    panel: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for source_row in source["by_horizon"][horizon]:
            meeting_id = source_row["event_meeting_id"]
            for arm in ARMS:
                panel.append({
                    "schema_version": SCHEMA + ":analysis-panel-row",
                    "backend_id": BACKEND_ID,
                    "convention": CONVENTION,
                    "horizon_months": horizon,
                    "horizon_label": f"FF{horizon}",
                    "arm": arm,
                    "paper_label": LABELS[arm],
                    "trading_date": str(source_row["trading_date"]),
                    "release_year": str(source_row["release_year"]),
                    "post_2011": int(str(source_row["trading_date"]) > POST_CUTOFF),
                    "is_minutes_release_event": meeting_id is not None,
                    "event_meeting_id": meeting_id,
                    "lag_meeting_id": source_row["lag_meeting_id"],
                    "news_shock": 0.0 if meeting_id is None else semantics["ns"][arm][str(meeting_id)],
                    "vix_log_percent_change": float(source_row["vix_log_percent_change"]),
                    "futures_return_log_percent": float(source_row["futures_return_log_percent"]),
                })
    return panel


def fit_interaction(
    y: np.ndarray,
    news: np.ndarray,
    vix: np.ndarray,
    years: Sequence[str],
    post: np.ndarray,
    *,
    hc1: bool,
) -> dict[str, float]:
    matrix, names = base.design_matrix(news, vix, years, post, interaction=True)
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
    if int(rank) != matrix.shape[1] or not np.all(np.isfinite(coefficients)):
        raise AnalysisError(f"rank-deficient/nonfinite interaction design: rank={rank}, k={matrix.shape[1]}")
    pre_index = names.index("news_shock")
    cross_index = names.index("post2011_x_news_shock")
    result = {
        "beta_pre": float(coefficients[pre_index]),
        "beta_interaction": float(coefficients[cross_index]),
        "beta_post": float(coefficients[pre_index] + coefficients[cross_index]),
    }
    if hc1:
        result["condition_number"] = float(np.linalg.cond(matrix))
        fitted = base.fit_ols_hc1(y, matrix, names)
        covariance = fitted["covariance"]
        post_variance = float(
            covariance[pre_index, pre_index]
            + covariance[cross_index, cross_index]
            + 2.0 * covariance[pre_index, cross_index]
        )
        result.update({
            "se_pre": float(fitted["standard_errors"][pre_index]),
            "se_interaction": float(fitted["standard_errors"][cross_index]),
            "se_post": math.sqrt(max(post_variance, 0.0)),
            "nobs": int(fitted["nobs"]),
            "r_squared": float(fitted["r_squared"]),
        })
    return result


def point_estimates(panel: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for arm in ARMS:
            rows = sorted(
                [row for row in panel if row["horizon_months"] == horizon and row["arm"] == arm],
                key=lambda row: str(row["trading_date"]),
            )
            fitted = fit_interaction(
                np.asarray([row["futures_return_log_percent"] for row in rows], dtype=np.float64),
                np.asarray([row["news_shock"] for row in rows], dtype=np.float64),
                np.asarray([row["vix_log_percent_change"] for row in rows], dtype=np.float64),
                [str(row["release_year"]) for row in rows],
                np.asarray([row["post_2011"] for row in rows], dtype=np.float64),
                hc1=True,
            )
            result.append({
                "schema_version": SCHEMA + ":coefficient-row",
                "backend_id": BACKEND_ID,
                "convention": CONVENTION,
                "specification": SPECIFICATION,
                "horizon_months": horizon,
                "horizon_label": f"FF{horizon}",
                "arm": arm,
                "paper_label": LABELS[arm],
                "nobs": fitted["nobs"],
                "beta_pre_basis_points": 100.0 * fitted["beta_pre"],
                "hc1_se_pre_basis_points": 100.0 * fitted["se_pre"],
                "beta_interaction_basis_points": 100.0 * fitted["beta_interaction"],
                "hc1_se_interaction_basis_points": 100.0 * fitted["se_interaction"],
                "beta_post_basis_points": 100.0 * fitted["beta_post"],
                "hc1_se_post_basis_points": 100.0 * fitted["se_post"],
                "r_squared": fitted["r_squared"],
                "condition_number": fitted["condition_number"],
                "covariance": "HC1 heteroskedasticity-robust",
            })
    return result


def make_bootstrap_plan(
    source: Mapping[str, Any], *, draws: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    pre_uniform = rng.random((draws, 7), dtype=np.float64)
    post_uniform = rng.random((draws, 4), dtype=np.float64)
    replicate_choices = rng.integers(0, 5, size=(draws, len(source["needed_ids"]), 5), dtype=np.int8)
    digest = hashlib.sha256()
    digest.update(pre_uniform.tobytes(order="C"))
    digest.update(post_uniform.tobytes(order="C"))
    digest.update(replicate_choices.tobytes(order="C"))
    return {
        "draws": draws,
        "seed": seed,
        "pre_uniform": pre_uniform,
        "post_uniform": post_uniform,
        "replicate_choices": replicate_choices,
        "plan_sha256": digest.hexdigest(),
    }


def sampled_years(plan: Mapping[str, Any], draw: int, horizon: int) -> tuple[str, ...]:
    pre = tuple(str(year) for year in range(2005 if horizon == 12 else 2004, 2011))
    post = tuple(str(year) for year in range(2012, 2016))
    pre_count = len(pre)
    selected_pre = tuple(pre[min(int(value * pre_count), pre_count - 1)] for value in plan["pre_uniform"][draw, :pre_count])
    selected_post = tuple(post[min(int(value * len(post)), len(post) - 1)] for value in plan["post_uniform"][draw])
    return selected_pre + ("2011",) + selected_post


def bootstrap_news(
    source: Mapping[str, Any], semantics: Mapping[str, Any], scores: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    needed_ids = tuple(source["needed_ids"])
    needed_index = {mid: index for index, mid in enumerate(needed_ids)}
    current_index = np.asarray([needed_index[mid] for mid in source["current_ids"]], dtype=np.int64)
    lag_index = np.asarray([needed_index[mid] for mid in source["lag_ids"]], dtype=np.int64)
    draws = int(plan["draws"])
    result: dict[str, np.ndarray] = {}
    official_rz = np.asarray([semantics["rz"]["official"][mid] for mid in needed_ids])
    official_ns = official_rz[current_index] - TADLE_GAMMA * official_rz[lag_index]
    result["official"] = np.broadcast_to(official_ns, (draws, len(current_index)))
    statement_z = np.asarray([semantics["statement_z"][mid] for mid in needed_ids])
    choices = np.asarray(plan["replicate_choices"], dtype=np.int64)
    for arm in GENERATED_ARMS:
        values = np.asarray([
            [scores["raw"]["generated"][arm][mid][rep] for rep in range(5)] for mid in needed_ids
        ], dtype=np.float64)
        expanded = np.broadcast_to(values, (draws, *values.shape))
        resampled_mean = np.take_along_axis(expanded, choices, axis=2).mean(axis=2)
        mean, sd = semantics["minute_scales"][arm]
        rz = (resampled_mean - mean) / sd - statement_z[None, :]
        result[arm] = rz[:, current_index] - TADLE_GAMMA * rz[:, lag_index]
    return result


def run_bootstrap(
    source: Mapping[str, Any], semantics: Mapping[str, Any], scores: Mapping[str, Any], plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    news_draws = bootstrap_news(source, semantics, scores, plan)
    current_index = {mid: index for index, mid in enumerate(source["current_ids"])}
    skeleton: dict[int, dict[str, Any]] = {}
    for horizon in HORIZONS:
        rows = source["by_horizon"][horizon]
        event_indexes = np.asarray([
            -1 if row["event_meeting_id"] is None else current_index[str(row["event_meeting_id"])]
            for row in rows
        ], dtype=np.int64)
        skeleton[horizon] = {
            "rows": rows,
            "event_indexes": event_indexes,
            "y": np.asarray([row["futures_return_log_percent"] for row in rows], dtype=np.float64),
            "vix": np.asarray([row["vix_log_percent_change"] for row in rows], dtype=np.float64),
            "post": np.asarray([row["post_2011"] for row in rows], dtype=np.float64),
            "years": tuple(str(row["release_year"]) for row in rows),
        }

    def demean_by_year(values: np.ndarray, years: Sequence[str]) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).copy()
        for year in sorted(set(years)):
            mask = np.asarray([value == year for value in years], dtype=bool)
            result[mask] -= result[mask].mean(axis=0)
        return result

    for data in skeleton.values():
        data["y_demeaned"] = demean_by_year(data["y"], data["years"])
        data["vix_demeaned"] = demean_by_year(data["vix"], data["years"])
        data["post_demeaned"] = demean_by_year(data["post"], data["years"])

    output: list[dict[str, Any]] = []
    for draw in range(int(plan["draws"])):
        draw_betas: dict[str, Any] = {}
        draw_years: dict[str, list[str]] = {}
        for horizon in HORIZONS:
            data = skeleton[horizon]
            years = sampled_years(plan, draw, horizon)
            draw_years[f"FF{horizon}"] = list(years)
            draw_betas[f"FF{horizon}"] = {}

            multiplicity = {year: years.count(year) for year in set(years)}
            weights = np.sqrt(np.asarray([multiplicity.get(year, 0) for year in data["years"]], dtype=np.float64))
            news_matrix = np.zeros((len(data["event_indexes"]), len(ARMS)), dtype=np.float64)
            mask = data["event_indexes"] >= 0
            for arm in ARMS:
                arm_index = ARMS.index(arm)
                news_matrix[mask, arm_index] = news_draws[arm][draw, data["event_indexes"][mask]]
            interaction_matrix = data["post"][:, None] * news_matrix
            paired_columns = np.empty((len(news_matrix), 2 * len(ARMS)), dtype=np.float64)
            for arm_index in range(len(ARMS)):
                paired_columns[:, 2 * arm_index] = news_matrix[:, arm_index]
                paired_columns[:, 2 * arm_index + 1] = interaction_matrix[:, arm_index]
            paired_demeaned = demean_by_year(paired_columns, data["years"])

            controls = np.column_stack((data["vix_demeaned"], data["post_demeaned"])) * weights[:, None]
            outcomes = np.column_stack((data["y_demeaned"], paired_demeaned)) * weights[:, None]
            control_coefficients, _, control_rank, _ = np.linalg.lstsq(controls, outcomes, rcond=None)
            if int(control_rank) != controls.shape[1]:
                raise AnalysisError("rank-deficient within-year bootstrap controls")
            residualized = outcomes - controls @ control_coefficients
            y_residual = residualized[:, 0]
            for arm_index, arm in enumerate(ARMS):
                columns = residualized[:, [1 + 2 * arm_index, 2 + 2 * arm_index]]
                gram = columns.T @ columns
                rhs = columns.T @ y_residual
                if np.linalg.matrix_rank(gram) != 2:
                    raise AnalysisError(f"rank-deficient bootstrap news terms: FF{horizon}/{arm}/{draw}")
                coefficients = np.linalg.solve(gram, rhs)
                draw_betas[f"FF{horizon}"][arm] = {
                    "pre_bp": 100.0 * float(coefficients[0]),
                    "interaction_bp": 100.0 * float(coefficients[1]),
                    "post_bp": 100.0 * float(coefficients.sum()),
                }
        output.append({
            "schema_version": SCHEMA + ":bootstrap-row",
            "draw_index": draw,
            "plan_sha256": plan["plan_sha256"],
            "sampled_years": draw_years,
            "betas": draw_betas,
        })
    return output


def summarize_bootstrap(
    points: Sequence[Mapping[str, Any]], draws: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    point = {(int(row["horizon_months"]), str(row["arm"])): row for row in points}
    estimands = {
        "pre": "beta_pre_basis_points",
        "interaction": "beta_interaction_basis_points",
        "post": "beta_post_basis_points",
    }
    rows: list[dict[str, Any]] = []
    for estimand, field in estimands.items():
        family = "primitive_pre_and_interaction_24" if estimand != "post" else "derived_post_marginal_12"
        for horizon in HORIZONS:
            official = float(point[(horizon, "official")][field])
            for arm in GENERATED_ARMS:
                estimate = float(point[(horizon, arm)][field]) - official
                values = np.asarray([
                    draw["betas"][f"FF{horizon}"][arm][f"{estimand}_bp"]
                    - draw["betas"][f"FF{horizon}"]["official"][f"{estimand}_bp"]
                    for draw in draws
                ], dtype=np.float64)
                low, high = beta_statistics.percentile_interval(values)
                rows.append({
                    "schema_version": SCHEMA + ":model-minus-official-row",
                    "backend_id": BACKEND_ID,
                    "convention": CONVENTION,
                    "specification": SPECIFICATION,
                    "family": family,
                    "estimand": estimand,
                    "horizon_months": horizon,
                    "horizon_label": f"FF{horizon}",
                    "arm": arm,
                    "paper_label": LABELS[arm],
                    "estimate_basis_points": estimate,
                    "ci_95_low_basis_points": low,
                    "ci_95_high_basis_points": high,
                    "bootstrap_p_raw": beta_statistics.two_sided_bootstrap_p_value(values),
                    "bootstrap_draws": len(draws),
                })
    for family in ("primitive_pre_and_interaction_24", "derived_post_marginal_12"):
        family_rows = [row for row in rows if row["family"] == family]
        adjusted = beta_statistics.holm_adjust([row["bootstrap_p_raw"] for row in family_rows])
        for row, value in zip(family_rows, adjusted, strict=True):
            row["holm_p"] = float(value)
    return rows


def fmt(value: float) -> str:
    return f"{value:+.4f}"


def render_tables(points: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]]) -> str:
    lookup = {(row["horizon_label"], row["arm"]): row for row in points}
    lines = [
        "# WGAN-Dictionary Tadle-Interaction Results",
        "",
        "Values are basis points; HC1 standard errors are in parentheses. Each cell is pre-2011 beta / interaction beta / post-2011 marginal beta.",
        "",
        "| Horizon | N | Official Minutes | Model chk-0 | Model chk-1 cp200 | Model chk-2 cp318 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("FF1", "FF3", "FF6", "FF12"):
        row0 = lookup[(label, "official")]
        cells = []
        for arm in ARMS:
            row = lookup[(label, arm)]
            cells.append(
                f"{fmt(row['beta_pre_basis_points'])} ({row['hc1_se_pre_basis_points']:.4f}) / "
                f"{fmt(row['beta_interaction_basis_points'])} ({row['hc1_se_interaction_basis_points']:.4f}) / "
                f"{fmt(row['beta_post_basis_points'])} ({row['hc1_se_post_basis_points']:.4f})"
            )
        lines.append(f"| {label} | {row0['nobs']:,} | " + " | ".join(cells) + " |")
    for estimand in ("pre", "interaction", "post"):
        lines.extend([
            "", f"## Model-minus-official contrasts: {estimand}", "",
            "| Horizon | Model | Difference (bp) | 95% paired bootstrap CI | Holm p |",
            "|---|---|---:|---:|---:|",
        ])
        for row in sorted(
            [item for item in contrasts if item["estimand"] == estimand],
            key=lambda item: (item["horizon_months"], item["arm"]),
        ):
            lines.append(
                f"| {row['horizon_label']} | {row['paper_label']} | {fmt(row['estimate_basis_points'])} | "
                f"[{fmt(row['ci_95_low_basis_points'])}, {fmt(row['ci_95_high_basis_points'])}] | "
                f"{row['holm_p']:.6f} |"
            )
    return "\n".join(lines) + "\n"


def render_report(
    points: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]], scores: Mapping[str, Any]
) -> str:
    significant = [row for row in contrasts if row["holm_p"] < 0.05]
    removals = scores["removal_rows"]
    return f"""# Tadle-Interaction Federal Funds Futures Diagnostic Using the WGAN Dictionary

## Result in brief

The requested calculation was completed with Tadle's post-August-8-2011 interaction equation and the historical WGAN Loughran--McDonald polarity contract. Across the 36 model-minus-official coefficient contrasts, {len(significant)} survive the prespecified Holm families at 5 percent. A failure to detect a difference is not evidence of statistical equivalence.

## Text measurement

The fixed scorer tokenizes with `{TOKEN_PATTERN.pattern}` and computes `(positive-negative)/max(1,positive+negative)`. The bound dictionary contains {scores['dictionary']['positive_unique_lexemes']:,} positive and {scores['dictionary']['negative_unique_lexemes']:,} negative unique lexemes. The official Minutes passage reproducing the current-meeting Statement is removed before scoring. All {len(removals)} current/lag documents passed the candidate-match gate; matching to the same-day Statement resolves meetings with multiple policy-statement passages.

## Regression

For each source and horizon, the full-sample point regression is `r = alpha + beta_pre NS + beta_interaction(post x NS) + gamma VIX + kappa post + calendar-year fixed effects + error`, where `post=1` after 2011-08-08. The reported post-period marginal slope is `beta_post=beta_pre+beta_interaction`. HC1 standard errors accompany full-sample OLS estimates.

## Bootstrap inference

The final inference layer uses 10,000 paired, regime-stratified calendar-year block draws. Each draw samples the available pre-2011 years with replacement, retains 2011 once, and samples 2012--2015 with replacement. It also resamples the five aligned generated documents within every current/lag meeting using model-shared replicate indices. Official Minutes and Statements remain deterministic, standardization scales remain fixed, and chronological lags are constructed before year blocks are duplicated. The 24 primitive pre/interaction contrasts form one Holm family; the 12 derived post-period marginal contrasts form a second family. This bootstrap is project-specific and conditions on the observed 2011 bridge year.

## Interpretation boundary

This is an exact implementation of the Tadle interaction *regression form* on the project's frozen market panel, not a literal replication of Tadle (2022). The WGAN dictionary is a financial-valence lexicon rather than Tadle's custom monetary-policy dictionary; generated documents contain eight concatenated Core8 paragraphs rather than full official Minutes; and the WRDS contract-continuation convention is project-specific. Generated documents were never released, so their coefficients are historical document-source association diagnostics rather than market effects.
"""


def validate(
    source: Mapping[str, Any], scores: Mapping[str, Any], panel: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]], draws: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]
) -> dict[str, Any]:
    expected_panel = 4 * sum(len(source["by_horizon"][h]) for h in HORIZONS)
    checks = {
        "dictionary_sha_matches": scores["dictionary"]["sha256"] == "e2d1328682bab7d2187684fb9f5420bb730401c9eefc00daf835edd203f4859d",
        "document_score_rows": len(scores["score_rows"]) == 1_428,
        "statement_removal_rows": len(scores["removal_rows"]) == 84,
        "all_statement_matches_above_threshold": all(row["similarity_ratio"] > 0.10 for row in scores["removal_rows"]),
        "analysis_panel_rows": len(panel) == expected_panel,
        "point_rows": len(points) == 16,
        "bootstrap_rows": len(draws) == int(plan["draws"]),
        "contrast_rows": len(contrasts) == 36,
        "post_identity_points": all(
            math.isclose(
                row["beta_post_basis_points"],
                row["beta_pre_basis_points"] + row["beta_interaction_basis_points"],
                rel_tol=0.0, abs_tol=1e-12,
            ) for row in points
        ),
        "post_identity_draws": all(
            math.isclose(
                values["post_bp"], values["pre_bp"] + values["interaction_bp"],
                rel_tol=0.0, abs_tol=1e-12,
            )
            for draw in draws for horizon in draw["betas"].values() for values in horizon.values()
        ),
        "finite_point_values": all(
            math.isfinite(float(value))
            for row in points
            for key, value in row.items()
            if isinstance(value, (int, float)) and key != "r_squared"
        ),
        "holm_values_in_unit_interval": all(0.0 <= row["holm_p"] <= 1.0 for row in contrasts),
    }
    status = "passed" if all(checks.values()) else "failed"
    if status != "passed":
        raise AnalysisError(f"validation failed: {checks}")
    return {
        "schema_version": SCHEMA + ":validation-report",
        "status": status,
        "checks": checks,
        "bootstrap_max_condition_number": max(
            0.0,
            *[
                abs(float(value))
                for row in points for key, value in row.items() if key == "condition_number"
            ],
        ),
        "limitations": [
            "The WGAN LM dictionary is not Tadle's original custom monetary-policy dictionary.",
            "The anchored bootstrap is conditional on the observed 2011 bridge year.",
            "Generated Core8 documents and official full Minutes have different information scope.",
            "The WRDS futures continuation convention is project-specific.",
        ],
    }


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"create-only output exists: {args.output_root}")
    if args.bootstrap_draws < 1:
        raise ValueError("bootstrap draws must be positive")
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_root.name}.staging.", dir=args.output_root.parent))
    completed = False
    try:
        source = load_source_skeleton()
        scores = build_scores(staging, source)
        semantics = construct_semantics(source, scores)
        write_json(staging / "sentiment/standardization_scales.json", {
            "schema_version": SCHEMA + ":standardization-scales",
            "sample_n": len(semantics["scale_ids"]),
            "ddof": 1,
            "statement": {"mean": semantics["statement_scale"][0], "sd": semantics["statement_scale"][1]},
            "minutes": {
                arm: {"mean": values[0], "sd": values[1]}
                for arm, values in semantics["minute_scales"].items()
            },
        })
        panel = build_panel(source, semantics)
        write_jsonl(staging / "estimation/analysis_panel.jsonl", panel)
        points = point_estimates(panel)
        write_jsonl(staging / "estimation/coefficient_estimates.jsonl", points)
        plan = make_bootstrap_plan(
            source, draws=args.bootstrap_draws, seed=args.bootstrap_seed
        )
        np.savez_compressed(
            staging / "estimation/bootstrap_plan_arrays.npz",
            pre_uniform=plan["pre_uniform"],
            post_uniform=plan["post_uniform"],
            replicate_choices=plan["replicate_choices"],
        )
        write_json(staging / "estimation/bootstrap_plan.json", {
            "schema_version": SCHEMA + ":bootstrap-plan",
            "draws": plan["draws"],
            "seed": plan["seed"],
            "plan_sha256": plan["plan_sha256"],
            "method": "regime-stratified 2011-anchored paired calendar-year pairs bootstrap",
            "year_blocks": {
                "FF1_FF3_FF6_pre": list(range(2004, 2011)),
                "FF12_pre": list(range(2005, 2011)),
                "anchor": [2011],
                "post": list(range(2012, 2016)),
            },
            "generated_document_resampling": "5 draws with replacement from K=5; indices shared across generated arms",
            "official_and_statement_resampling": False,
            "standardization_scales_fixed": True,
        })
        draws = run_bootstrap(source, semantics, scores, plan)
        write_jsonl(staging / "estimation/bootstrap_draws.jsonl", draws)
        contrasts = summarize_bootstrap(points, draws)
        write_jsonl(staging / "estimation/model_minus_official_contrasts.jsonl", contrasts)
        write_text(staging / "result_tables.md", render_tables(points, contrasts))
        write_text(staging / "technical_report.md", render_report(points, contrasts, scores))
        validation = validate(source, scores, panel, points, draws, contrasts, plan)
        write_json(staging / "validation_report.json", validation)
        significant = [row for row in contrasts if row["holm_p"] < 0.05]
        manifest = {
            "schema_version": SCHEMA + ":release-manifest",
            "created_at_utc": utc_now(),
            "status": "complete",
            "exact_tadle_regression_form": True,
            "exact_tadle_2022_replication": False,
            "interpretation": "historical reduced-form document-source coefficient diagnostic",
            "formula": {
                "document_score": "(P-N)/max(1,P+N)",
                "relative_sentiment": "RZ[a,t]=Z_minutes[a,t]-Z_statement[t]",
                "news_shock": "NS[a,t]=RZ[a,t]-0.368*RZ[a,t-1]",
                "regression": (
                    "r[h,t]=alpha[h]+beta_pre[h]*NS[a,t]+beta_interaction[h]*(post[t]*NS[a,t])"
                    "+gamma[h]*VIX[t]+kappa[h]*post[t]+calendar-year FE+error[h,t]"
                ),
                "post_marginal": "beta_post=beta_pre+beta_interaction",
            },
            "sample": {
                "date_start": "2004-12-01",
                "date_end": "2015-04-30",
                "standardization_meetings": 83,
                "estimable_events": 82,
                "nobs": {"FF1": 2613, "FF3": 2612, "FF6": 2612, "FF12": 2338},
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "environment": {
                    key: os.environ.get(key)
                    for key in ("PYTHONNOUSERSITE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
                },
            },
            "results": {
                "holm_significant_contrasts": len(significant),
                "total_contrasts": len(contrasts),
            },
            "inputs": {
                "implementation": binding(IMPLEMENTATION),
                "source_release_manifest": binding(SOURCE_MANIFEST),
                "source_analysis_panel": binding(SOURCE_PANEL),
                "statement_documents": binding(STATEMENT_DOCUMENTS),
                "official_documents": binding(OFFICIAL_DOCUMENTS),
                "generated_documents": binding(GENERATED_DOCUMENTS),
                "wgan_dictionary": binding(DICTIONARY),
            },
            "artifacts": {
                "dictionary_manifest": binding(staging / "sentiment/dictionary_manifest.json", relative_to=staging),
                "document_scores": binding(staging / "sentiment/document_scores.jsonl", relative_to=staging, rows=len(scores["score_rows"])),
                "statement_removal_ledger": binding(staging / "sentiment/official_statement_removal_ledger.jsonl", relative_to=staging, rows=len(scores["removal_rows"])),
                "standardization_scales": binding(staging / "sentiment/standardization_scales.json", relative_to=staging),
                "analysis_panel": binding(staging / "estimation/analysis_panel.jsonl", relative_to=staging, rows=len(panel)),
                "coefficient_estimates": binding(staging / "estimation/coefficient_estimates.jsonl", relative_to=staging, rows=len(points)),
                "bootstrap_plan": binding(staging / "estimation/bootstrap_plan.json", relative_to=staging),
                "bootstrap_plan_arrays": binding(staging / "estimation/bootstrap_plan_arrays.npz", relative_to=staging),
                "bootstrap_draws": binding(staging / "estimation/bootstrap_draws.jsonl", relative_to=staging, rows=len(draws)),
                "contrasts": binding(staging / "estimation/model_minus_official_contrasts.jsonl", relative_to=staging, rows=len(contrasts)),
                "result_tables": binding(staging / "result_tables.md", relative_to=staging),
                "technical_report": binding(staging / "technical_report.md", relative_to=staging),
                "validation_report": binding(staging / "validation_report.json", relative_to=staging),
            },
            "limitations": validation["limitations"],
        }
        write_json(staging / "manifest.json", manifest)
        os.replace(staging, args.output_root)
        completed = True
        print(canonical({
            "status": "complete",
            "output_root": str(args.output_root.resolve()),
            "holm_significant_contrasts": len(significant),
        }))
        return 0
    finally:
        if not completed and staging.exists():
            print(f"Partial staging retained for audit: {staging}", file=os.sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
