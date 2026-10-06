#!/usr/bin/env python3
"""Deeply validate the sealed Tadle-form FF futures diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from jobs.eval import run_tadle_form_ff_futures_v1 as runner


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE = (
    ROOT / "output/evaluation/main/tadle_form_official_generated_ff_futures_2004_2015_v1"
)
DEFAULT_OUTPUT = (
    ROOT / "output/evaluation/main/tadle_form_official_generated_ff_futures_2004_2015_v1_deep_validation_v3"
)
SCHEMA = "tadle-form-ff-futures-deep-validation-v3"
EXPECTED_RELEASE_MANIFEST_SHA256 = "eeb29722e95903dadedfa9e496b6a8e017ef0b731d8e5a67befc6ffd198de44f"


class ValidationError(RuntimeError):
    """A sealed-release replay check failed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_new_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)


def reject_json_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant is forbidden: {value}")


def strict_json_loads(value: str) -> Any:
    return json.loads(value, parse_constant=reject_json_constant)


def read_json(path: Path) -> Any:
    return strict_json_loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [strict_json_loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValidationError(f"non-object JSONL row: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def assert_finite_close(name: str, actual: float, expected: float, *, tolerance: float = 1e-12) -> float:
    actual_value = float(actual)
    expected_value = float(expected)
    if not math.isfinite(actual_value) or not math.isfinite(expected_value):
        raise ValidationError(f"non-finite numeric value in {name}: {actual_value}, {expected_value}")
    error = abs(actual_value - expected_value)
    if error > tolerance:
        raise ValidationError(f"numeric replay mismatch in {name}: error={error}")
    return error


def independent_holm(values: Sequence[float]) -> list[float]:
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in converted):
        raise ValidationError("invalid p-value supplied to Holm adjustment")
    ordered = sorted(enumerate(converted), key=lambda item: item[1])
    result = [0.0] * len(ordered)
    running = 0.0
    total = len(ordered)
    for rank, (original, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        result[original] = running
    return result


def sign_p(values: np.ndarray) -> float:
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValidationError("non-finite or non-vector bootstrap contrast distribution")
    tail = min(int(np.count_nonzero(values <= 0)), int(np.count_nonzero(values >= 0)))
    return min(1.0, 2.0 * (tail + 1.0) / (len(values) + 1.0))


def count_nonempty_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def verify_bound(
    path: Path,
    binding: Mapping[str, Any],
    *,
    allowed_root: Path,
) -> dict[str, Any]:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ValidationError(f"bound file missing: {path}")
    if not resolved.is_relative_to(allowed_root.resolve()):
        raise ValidationError(f"bound path escapes the allowed root: {path}")
    actual_hash = sha256_file(resolved)
    if resolved.stat().st_size != int(binding["bytes"]) or actual_hash != binding["sha256"]:
        raise ValidationError(f"bound file changed: {path}")
    if "rows" in binding:
        actual_rows = count_nonempty_lines(resolved)
        if actual_rows != int(binding["rows"]):
            raise ValidationError(f"bound row count changed: {path}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual_hash,
        **({"rows": int(binding["rows"])} if "rows" in binding else {}),
    }


def verify_statement_replay(
    release: Path,
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    source_rows = read_jsonl(release / "statements/source_manifest.jsonl")
    document_rows = read_jsonl(release / "statements/documents.jsonl")
    sources = {row["meeting_id"]: row for row in source_rows}
    documents = {row["meeting_id"]: row for row in document_rows}
    if (
        len(source_rows) != len(sources)
        or len(document_rows) != len(documents)
        or set(sources) != set(documents)
        or len(sources) != 84
    ):
        raise ValidationError("Statement source/document coverage drift")
    for mid, source in sources.items():
        raw_path = release / source["raw_path"]
        if sha256_file(raw_path) != source["raw_sha256"]:
            raise ValidationError(f"Statement raw hash mismatch: {mid}")
        text, metadata = runner.clean_statement_html(raw_path.read_bytes())
        document = documents[mid]
        if (
            source["schema_version"] != runner.SCHEMA + ":statement-source-row"
            or document["schema_version"] != runner.SCHEMA + ":statement-document-row"
            or document["document_id"] != f"statement::{mid}"
            or document["meeting_end_date"] != mid
            or document["source_raw_sha256"] != source["raw_sha256"]
            or document["document_text_sha256"] != source["text_sha256"]
            or source["no_silent_truncation"] is not True
            or document["no_silent_truncation"] is not True
            or text != document["document_text"]
            or metadata["text_sha256"] != source["text_sha256"]
            or int(metadata["characters"]) != int(source["characters"])
            or int(metadata["whitespace_words"]) != int(source["whitespace_words"])
            or metadata["selector"] != source["selector"]
            or int(metadata["removed_nodes"]) != int(source["removed_nodes"])
        ):
            raise ValidationError(f"Statement clean-body replay mismatch: {mid}")

    backend_summary: dict[str, Any] = {}
    for backend_id in runner.BACKEND_IDS:
        tokenizer = AutoTokenizer.from_pretrained(
            runner.official_runner.BACKENDS[backend_id], local_files_only=True, use_fast=True
        )
        window_rows = read_jsonl(release / "statements/sentiment" / backend_id / "window_scores.jsonl")
        score_row_list = read_jsonl(
            release / "statements/sentiment" / backend_id / "document_scores.jsonl"
        )
        score_rows = {row["document_id"]: row for row in score_row_list}
        if len(score_row_list) != len(score_rows) or set(score_rows) != {
            str(row["document_id"]) for row in documents.values()
        }:
            raise ValidationError(f"Statement document-score key coverage drift: {backend_id}")
        by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
        window_keys: set[tuple[str, int]] = set()
        for row in window_rows:
            key = (str(row["document_id"]), int(row["window_index"]))
            if key in window_keys:
                raise ValidationError(f"duplicate Statement window row: {backend_id}/{key}")
            window_keys.add(key)
            by_document[str(row["document_id"])].append(row)
        maximum_error = 0.0
        for document in documents.values():
            doc_id = str(document["document_id"])
            specs = runner.official_runner.window_specs(tokenizer, str(document["document_text"]))
            saved = sorted(by_document[doc_id], key=lambda row: int(row["window_index"]))
            if len(specs) != len(saved):
                raise ValidationError(f"Statement window count mismatch: {backend_id}/{doc_id}")
            weighted = np.zeros(3, dtype=np.float64)
            total_weight = 0.0
            for spec, row in zip(specs, saved, strict=True):
                checks = (
                    int(row["token_start"]) == int(spec["token_start"]),
                    int(row["token_end"]) == int(spec["token_end"]),
                    row["content_token_ids_sha256"] == spec["content_token_ids_sha256"],
                    math.isclose(
                        float(row["aggregation_weight"]),
                        float(spec["aggregation_weight"]),
                        abs_tol=1e-12,
                        rel_tol=0,
                    ),
                    row["schema_version"] == runner.SCHEMA + ":statement-window-score-row",
                    row["backend_id"] == backend_id,
                    row["meeting_id"] == document["meeting_id"],
                    int(row["content_token_count"])
                    == int(spec["token_end"] - spec["token_start"]),
                )
                if not all(checks):
                    raise ValidationError(f"Statement token-window replay mismatch: {backend_id}/{doc_id}")
                weight = float(row["aggregation_weight"])
                row_probabilities = np.asarray(row["probabilities"], dtype=np.float64)
                if (
                    row_probabilities.shape != (3,)
                    or not np.all(np.isfinite(row_probabilities))
                    or not math.isclose(float(row_probabilities.sum()), 1.0, abs_tol=2e-6, rel_tol=0)
                ):
                    raise ValidationError(f"invalid Statement window probabilities: {backend_id}/{doc_id}")
                signed = read_json(
                    runner.official_runner.BACKENDS[backend_id] / "model_manifest.json"
                )["signed_score"]
                expected_window_score = (
                    row_probabilities[int(signed["positive_index"])]
                    - row_probabilities[int(signed["negative_index"])]
                )
                assert_finite_close(
                    f"Statement window score {backend_id}/{doc_id}",
                    row["score"],
                    expected_window_score,
                )
                weighted += weight * row_probabilities
                total_weight += weight
            probabilities = weighted / total_weight
            score_row = score_rows[doc_id]
            target = np.asarray(score_row["probabilities"], dtype=np.float64)
            if target.shape != (3,) or not np.all(np.isfinite(target)):
                raise ValidationError(f"invalid Statement document probabilities: {backend_id}/{doc_id}")
            probability_error = float(np.max(np.abs(probabilities - target)))
            if not math.isfinite(probability_error):
                raise ValidationError(f"non-finite Statement aggregation error: {backend_id}/{doc_id}")
            maximum_error = max(maximum_error, probability_error)
            if not np.allclose(probabilities, target, atol=1e-12, rtol=0):
                raise ValidationError(f"Statement probability aggregation mismatch: {backend_id}/{doc_id}")
            expected_document_score = (
                target[int(signed["positive_index"])]
                - target[int(signed["negative_index"])]
            )
            assert_finite_close(
                f"Statement document score {backend_id}/{doc_id}",
                score_row["score"],
                expected_document_score,
            )
            if (
                score_row["schema_version"] != runner.SCHEMA + ":statement-document-score-row"
                or score_row["backend_id"] != backend_id
                or score_row["meeting_id"] != document["meeting_id"]
                or score_row["document_text_sha256"] != document["document_text_sha256"]
                or int(score_row["window_count"]) != len(specs)
                or int(score_row["content_token_count"]) != int(specs[-1]["token_end"])
                or score_row["no_silent_truncation"] is not True
            ):
                raise ValidationError(f"Statement document-score metadata drift: {backend_id}/{doc_id}")
        backend_summary[backend_id] = {
            "documents": len(score_rows),
            "windows": len(window_rows),
            "maximum_probability_replay_error": maximum_error,
        }

    # Re-run the frozen model forward passes in an isolated temporary output
    # and compare every window- and document-level probability.  This is
    # deliberately separate from the token-window/aggregation replay above.
    with tempfile.TemporaryDirectory(prefix="tadle-statement-forward-replay-") as temporary:
        temporary_root = Path(temporary)
        for backend_id in runner.BACKEND_IDS:
            runner.score_statement_backend(
                temporary_root,
                list(documents.values()),
                backend_id,
                device=device,
                batch_size=batch_size,
            )
            saved_windows = {
                (str(row["document_id"]), int(row["window_index"])): row
                for row in read_jsonl(
                    release / "statements/sentiment" / backend_id / "window_scores.jsonl"
                )
            }
            replayed_windows = {
                (str(row["document_id"]), int(row["window_index"])): row
                for row in read_jsonl(
                    temporary_root / "statements/sentiment" / backend_id / "window_scores.jsonl"
                )
            }
            saved_documents = {
                str(row["document_id"]): row
                for row in read_jsonl(
                    release / "statements/sentiment" / backend_id / "document_scores.jsonl"
                )
            }
            replayed_documents = {
                str(row["document_id"]): row
                for row in read_jsonl(
                    temporary_root / "statements/sentiment" / backend_id / "document_scores.jsonl"
                )
            }
            if set(saved_windows) != set(replayed_windows) or set(saved_documents) != set(replayed_documents):
                raise ValidationError(f"Statement forward-pass coverage mismatch: {backend_id}")
            maximum_forward_error = 0.0
            for key, saved in saved_windows.items():
                replayed = replayed_windows[key]
                error = float(np.max(np.abs(
                    np.asarray(saved["probabilities"], dtype=np.float64)
                    - np.asarray(replayed["probabilities"], dtype=np.float64)
                )))
                if not math.isfinite(error):
                    raise ValidationError(f"non-finite Statement forward error: {backend_id}/{key}")
                maximum_forward_error = max(
                    maximum_forward_error,
                    error,
                    assert_finite_close(
                        f"Statement forward window score {backend_id}/{key}",
                        saved["score"],
                        replayed["score"],
                        tolerance=1e-7,
                    ),
                )
            for key, saved in saved_documents.items():
                replayed = replayed_documents[key]
                error = float(np.max(np.abs(
                    np.asarray(saved["probabilities"], dtype=np.float64)
                    - np.asarray(replayed["probabilities"], dtype=np.float64)
                )))
                if not math.isfinite(error):
                    raise ValidationError(f"non-finite Statement forward error: {backend_id}/{key}")
                maximum_forward_error = max(
                    maximum_forward_error,
                    error,
                    assert_finite_close(
                        f"Statement forward document score {backend_id}/{key}",
                        saved["score"],
                        replayed["score"],
                        tolerance=1e-7,
                    ),
                )
            if maximum_forward_error > 1e-7:
                raise ValidationError(
                    f"Statement model-forward replay mismatch: {backend_id}, error={maximum_forward_error}"
                )
            backend_summary[backend_id]["maximum_model_forward_replay_error"] = maximum_forward_error
    return {
        "sources": len(sources),
        "cleaned_documents": len(documents),
        "backends": backend_summary,
    }


def verify_futures(release: Path) -> dict[str, Any]:
    rows = read_jsonl(release / "futures/futures_return_ledger.jsonl")
    actual_keys = {
        (str(row["convention"]), int(row["horizon_months"]), str(row["trading_date"]))
        for row in rows
    }
    if len(rows) != len(actual_keys):
        raise ValidationError("duplicate futures construction rows")
    # The ledger is at the WRDS trading-date grain, not calendar-day grain;
    # validate the exact count/key dimensions below rather than requiring the
    # superset of calendar dates constructed above.
    if len(rows) != 20_968 or not all(key[0] in runner.CONVENTIONS and key[1] in runner.HORIZONS for key in actual_keys):
        raise ValidationError("futures construction key coverage drift")
    metadata = pd.read_csv(runner.WRDS_METADATA, parse_dates=["lasttrddate"])
    metadata["contract_month"] = metadata["lasttrddate"].dt.to_period("M")
    by_code = {int(row.futcode): row for row in metadata.itertuples(index=False)}
    maximum_return_error = 0.0
    for row in rows:
        if row["status"] != "complete":
            continue
        expected = 100.0 * math.log(float(row["current_settlement"]) / float(row["previous_valid_settlement"]))
        error = assert_finite_close(
            "futures return formula",
            row["return_log_percent"],
            expected,
            tolerance=1e-14,
        )
        maximum_return_error = max(maximum_return_error, error)
        contract = by_code[int(row["futcode"])]
        date = pd.Timestamp(row["trading_date"])
        horizon = int(row["horizon_months"])
        if row["convention"] == "calendar_month_offset":
            if contract.contract_month != date.to_period("M") + horizon:
                raise ValidationError("calendar-month contract mapping mismatch")
        else:
            eligible = metadata.loc[metadata["lasttrddate"] >= date].sort_values(["lasttrddate", "futcode"])
            if int(eligible.iloc[horizon - 1]["futcode"]) != int(row["futcode"]):
                raise ValidationError("live-rank contract mapping mismatch")
    return {
        "ledger_rows": len(rows),
        "complete_rows": sum(row["status"] == "complete" for row in rows),
        "maximum_return_formula_error": maximum_return_error,
    }


def independent_design(frame: pd.DataFrame, interaction: bool) -> tuple[np.ndarray, list[str]]:
    columns = [np.ones(len(frame)), frame["news_shock"].to_numpy(float), frame["vix_log_percent_change"].to_numpy(float)]
    names = ["intercept", "news_shock", "vix"]
    if interaction:
        post = frame["post_2011"].to_numpy(float)
        columns.extend((post * frame["news_shock"].to_numpy(float), post))
        names.extend(("post2011_x_news_shock", "post2011"))
    levels = sorted(frame["release_year"].astype(str).unique())
    for level in levels[1:]:
        columns.append((frame["release_year"].astype(str) == level).to_numpy(float))
        names.append(f"year_{level}")
    return np.column_stack(columns), names


def independent_hc1(y: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beta = np.linalg.lstsq(matrix, y, rcond=None)[0]
    residual = y - matrix @ beta
    nobs, k = matrix.shape
    bread = np.linalg.inv(matrix.T @ matrix)
    covariance = (nobs / (nobs - k)) * bread @ (matrix.T @ ((residual ** 2)[:, None] * matrix)) @ bread
    centered = y - y.mean()
    tss = float(centered @ centered)
    rss = float(residual @ residual)
    return beta, covariance, float(1.0 - rss / tss) if tss > 0 else float("nan")


def verify_point_estimates(release: Path) -> dict[str, Any]:
    panel_rows = read_jsonl(release / "estimation/analysis_panel.jsonl")
    panel = pd.DataFrame(panel_rows)
    panel_keys = {
        (
            row["backend_id"],
            row["convention"],
            int(row["horizon_months"]),
            row["arm"],
            row["trading_date"],
        )
        for row in panel_rows
    }
    if len(panel_rows) != 163_200 or len(panel_keys) != len(panel_rows):
        raise ValidationError("analysis-panel row/key coverage drift")
    for column in (
        "news_shock",
        "vix_log_percent_change",
        "futures_return_log_percent",
        "post_2011",
    ):
        values = pd.to_numeric(panel[column], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(values)):
            raise ValidationError(f"non-finite analysis-panel values: {column}")
    saved = read_jsonl(release / "estimation/coefficient_estimates.jsonl")
    expected_keys = {
        (backend_id, convention, horizon, arm, specification)
        for backend_id in runner.BACKEND_IDS
        for convention in runner.CONVENTIONS
        for horizon in runner.HORIZONS
        for arm in runner.ARMS
        for specification in ("basic_tadle_form", "post2011_interaction_tadle_form")
    }
    actual_keys = {
        (
            row["backend_id"],
            row["convention"],
            int(row["horizon_months"]),
            row["arm"],
            row["specification"],
        )
        for row in saved
    }
    if len(saved) != 128 or len(actual_keys) != len(saved) or actual_keys != expected_keys:
        raise ValidationError("point-estimate Cartesian key coverage drift")
    maximum_beta_error = 0.0
    maximum_se_error = 0.0
    for row in saved:
        frame = panel.loc[
            (panel["backend_id"] == row["backend_id"])
            & (panel["convention"] == row["convention"])
            & (panel["horizon_months"] == int(row["horizon_months"]))
            & (panel["arm"] == row["arm"])
        ].sort_values("trading_date")
        interaction = row["specification"] == "post2011_interaction_tadle_form"
        matrix, names = independent_design(frame, interaction)
        y = frame["futures_return_log_percent"].to_numpy(float)
        beta, covariance, r_squared = independent_hc1(y, matrix)
        index = names.index("news_shock")
        beta_error = assert_finite_close(
            "point beta_news_shock", row["beta_news_shock"], beta[index]
        )
        news_se = math.sqrt(float(covariance[index, index]))
        se_error = assert_finite_close(
            "point hc1_se_news_shock", row["hc1_se_news_shock"], news_se
        )
        maximum_beta_error = max(maximum_beta_error, beta_error)
        maximum_se_error = max(maximum_se_error, se_error)
        assert_finite_close("point beta basis-point mirror", row["beta_news_shock_basis_points"], 100.0 * beta[index])
        assert_finite_close("point SE basis-point mirror", row["hc1_se_news_shock_basis_points"], 100.0 * news_se)
        assert_finite_close("point r_squared", row["r_squared"], r_squared)
        if (
            int(row["nobs"]) != len(frame)
            or row["schema_version"] != runner.SCHEMA + ":coefficient-row"
            or row["horizon_label"] != f"FF{int(row['horizon_months'])}"
            or row["paper_label"] != runner.PAPER_LABELS[row["arm"]]
            or row["covariance"] != "HC1 heteroskedasticity-robust"
        ):
            raise ValidationError("point-estimate metadata drift")
        if interaction:
            cross = names.index("post2011_x_news_shock")
            cross_se = math.sqrt(float(covariance[cross, cross]))
            post_beta = float(beta[index] + beta[cross])
            post_variance = float(covariance[index, index] + covariance[cross, cross] + 2 * covariance[index, cross])
            post_se = math.sqrt(max(post_variance, 0.0))
            interaction_checks = (
                ("interaction beta", row["beta_post2011_x_news_shock"], beta[cross]),
                ("interaction SE", row["hc1_se_post2011_x_news_shock"], cross_se),
                ("interaction beta bp", row["beta_post2011_x_news_shock_basis_points"], 100.0 * beta[cross]),
                ("interaction SE bp", row["hc1_se_post2011_x_news_shock_basis_points"], 100.0 * cross_se),
                ("post-2011 beta", row["beta_news_shock_post2011"], post_beta),
                ("post-2011 SE", row["hc1_se_news_shock_post2011"], post_se),
                ("post-2011 beta bp", row["beta_news_shock_post2011_basis_points"], 100.0 * post_beta),
                ("post-2011 SE bp", row["hc1_se_news_shock_post2011_basis_points"], 100.0 * post_se),
            )
            for name, actual, expected in interaction_checks:
                assert_finite_close(name, actual, expected)
        elif any(
            row.get(name) is not None
            for name in (
                "beta_post2011_x_news_shock",
                "hc1_se_post2011_x_news_shock",
                "beta_news_shock_post2011",
                "hc1_se_news_shock_post2011",
            )
        ):
            raise ValidationError("basic specification violates interaction-null contract")
    return {
        "coefficient_rows": len(saved),
        "maximum_beta_error": maximum_beta_error,
        "maximum_hc1_se_error": maximum_se_error,
    }


def regenerate_plan(release: Path) -> dict[str, Any]:
    public = read_json(release / "estimation/bootstrap_plan.json")
    plan_sha = public.pop("plan_sha256")
    rng = np.random.default_rng(int(public["seed"]))
    year_indexes = rng.integers(
        0,
        len(public["calendar_year_blocks"]),
        size=(int(public["draws"]), int(public["blocks_per_draw"])),
        dtype=np.int16,
    )
    choices = rng.integers(
        0,
        5,
        size=(int(public["draws"]), len(public["needed_meetings"]), 5),
        dtype=np.int8,
    )
    computed = hashlib.sha256(
        (
            canonical(public)
            + hashlib.sha256(year_indexes.tobytes(order="C")).hexdigest()
            + hashlib.sha256(choices.tobytes(order="C")).hexdigest()
        ).encode("utf-8")
    ).hexdigest()
    if computed != plan_sha:
        raise ValidationError("bootstrap plan hash replay mismatch")
    return {
        "public": public,
        "plan_sha256": plan_sha,
        "year_indexes": year_indexes,
        "choices": choices,
    }


def independent_news_shock_draws(
    semantic: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    draws = int(plan["contract"]["draws"])
    needed_ids = tuple(plan["needed_ids"])
    needed_index = {mid: index for index, mid in enumerate(needed_ids)}
    current_index = np.asarray(
        [needed_index[mid] for mid in plan["current_ids"]], dtype=np.int64
    )
    lag_index = np.asarray(
        [needed_index[mid] for mid in semantic["lag_ids"]], dtype=np.int64
    )
    statement_z = np.asarray(
        [semantic["statement_z"][mid] for mid in needed_ids], dtype=np.float64
    )
    choices = np.asarray(plan["replicate_choices"], dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for arm in runner.ARMS:
        mean, standard_deviation = semantic["minute_scales"][arm]
        if arm == "official":
            minute_z = np.asarray(
                [
                    (semantic["point_minutes"][arm][mid] - mean) / standard_deviation
                    for mid in needed_ids
                ],
                dtype=np.float64,
            )
            relative = minute_z - statement_z
            news = (
                relative[current_index]
                - runner.TADLE_GAMMA * relative[lag_index]
            )
            result[arm] = np.broadcast_to(news, (draws, len(current_index)))
        else:
            raw = np.asarray(
                [
                    [semantic["minutes"]["generated"][arm][mid][rep] for rep in range(5)]
                    for mid in needed_ids
                ],
                dtype=np.float64,
            )
            expanded = np.broadcast_to(raw, (draws, *raw.shape))
            resampled_mean = np.take_along_axis(expanded, choices, axis=2).mean(axis=2)
            relative = (resampled_mean - mean) / standard_deviation - statement_z[None, :]
            result[arm] = (
                relative[:, current_index]
                - runner.TADLE_GAMMA * relative[:, lag_index]
            )
        if not np.all(np.isfinite(result[arm])):
            raise ValidationError(f"non-finite independently reconstructed news shocks: {arm}")
    return result


def independent_bootstrap_backend(
    semantic: Mapping[str, Any],
    panel_rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Re-estimate every draw with year-FE sufficient statistics.

    Resampling a complete calendar year is algebraically equivalent to
    weighting every row in that year by its draw multiplicity.  Partialling
    out year fixed effects therefore reduces each draw to a two-regressor
    weighted system for demeaned news and VIX.  This implementation is
    independent of the producer's repeated-row ``lstsq`` loop.
    """

    backend_id = str(semantic["backend_id"])
    years = tuple(plan["years"])
    year_indexes = np.asarray(plan["sampled_year_indexes"], dtype=np.int64)
    year_counts = np.column_stack(
        [np.count_nonzero(year_indexes == index, axis=1) for index in range(len(years))]
    ).astype(np.float64)
    draws = int(plan["contract"]["draws"])
    news_draws = independent_news_shock_draws(semantic, plan)
    current_ids = tuple(plan["current_ids"])
    current_index = {mid: index for index, mid in enumerate(current_ids)}
    beta_arrays: dict[str, dict[str, np.ndarray]] = {}

    for horizon in runner.HORIZONS:
        subset = sorted(
            [
                row
                for row in panel_rows
                if row["backend_id"] == backend_id
                and row["convention"] == runner.PRIMARY_CONVENTION
                and int(row["horizon_months"]) == horizon
                and row["arm"] == "official"
            ],
            key=lambda row: str(row["trading_date"]),
        )
        y = np.asarray([row["futures_return_log_percent"] for row in subset], dtype=np.float64)
        vix = np.asarray([row["vix_log_percent_change"] for row in subset], dtype=np.float64)
        row_year_indexes = np.asarray([years.index(str(row["release_year"])) for row in subset])
        event_indexes = np.asarray(
            [
                -1
                if row["event_meeting_id"] is None
                else current_index[str(row["event_meeting_id"])]
                for row in subset
            ],
            dtype=np.int64,
        )
        vv = np.zeros(len(years), dtype=np.float64)
        vy = np.zeros(len(years), dtype=np.float64)
        centered_vix = np.zeros(len(subset), dtype=np.float64)
        centered_y = np.zeros(len(subset), dtype=np.float64)
        rows_per_year = np.zeros(len(years), dtype=np.int64)
        for year_index in range(len(years)):
            positions = np.flatnonzero(row_year_indexes == year_index)
            rows_per_year[year_index] = len(positions)
            if not len(positions):
                continue
            centered_vix[positions] = vix[positions] - vix[positions].mean()
            centered_y[positions] = y[positions] - y[positions].mean()
            vv[year_index] = centered_vix[positions] @ centered_vix[positions]
            vy[year_index] = centered_vix[positions] @ centered_y[positions]
        a22 = year_counts @ vv
        b2 = year_counts @ vy
        beta_arrays[f"FF{horizon}"] = {}
        for arm in runner.ARMS:
            nn = np.zeros((draws, len(years)), dtype=np.float64)
            nv = np.zeros((draws, len(years)), dtype=np.float64)
            ny = np.zeros((draws, len(years)), dtype=np.float64)
            for year_index in range(len(years)):
                positions = np.flatnonzero(
                    (row_year_indexes == year_index) & (event_indexes >= 0)
                )
                if not len(positions):
                    continue
                values = news_draws[arm][:, event_indexes[positions]]
                totals = values.sum(axis=1)
                nn[:, year_index] = (
                    np.square(values).sum(axis=1)
                    - np.square(totals) / float(rows_per_year[year_index])
                )
                nv[:, year_index] = values @ centered_vix[positions]
                ny[:, year_index] = values @ centered_y[positions]
            a11 = np.sum(year_counts * nn, axis=1)
            a12 = np.sum(year_counts * nv, axis=1)
            b1 = np.sum(year_counts * ny, axis=1)
            denominator = a11 * a22 - np.square(a12)
            if np.any(~np.isfinite(denominator)) or np.any(np.abs(denominator) < 1e-20):
                raise ValidationError(f"rank-deficient independent bootstrap system: FF{horizon}/{arm}")
            betas = (b1 * a22 - b2 * a12) / denominator
            if not np.all(np.isfinite(betas)):
                raise ValidationError(f"non-finite independent bootstrap betas: FF{horizon}/{arm}")
            beta_arrays[f"FF{horizon}"][arm] = betas

    result: list[dict[str, Any]] = []
    for draw in range(draws):
        sampled_years = [years[int(index)] for index in year_indexes[draw]]
        result.append({
            "schema_version": runner.SCHEMA + ":bootstrap-row",
            "backend_id": backend_id,
            "draw_index": draw,
            "plan_sha256": plan["plan_sha256"],
            "sampled_calendar_years": sampled_years,
            "specification": "basic_tadle_form",
            "convention": runner.PRIMARY_CONVENTION,
            "betas": {
                label: {arm: float(values[arm][draw]) for arm in runner.ARMS}
                for label, values in beta_arrays.items()
            },
        })
    return result


def replay_bootstrap_draws(
    release: Path,
    plan: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    sample = runner.load_sample()
    replayed_plan = runner.build_bootstrap_plan(
        sample,
        draws=int(plan["public"]["draws"]),
        seed=int(plan["public"]["seed"]),
    )
    if replayed_plan["plan_sha256"] != plan["plan_sha256"]:
        raise ValidationError("runner bootstrap plan replay mismatch")
    if not np.array_equal(replayed_plan["sampled_year_indexes"], plan["year_indexes"]):
        raise ValidationError("runner year-block draw replay mismatch")
    if not np.array_equal(replayed_plan["replicate_choices"], plan["choices"]):
        raise ValidationError("runner replicate draw replay mismatch")

    futures_rows = read_jsonl(release / "futures/futures_return_ledger.jsonl")
    panel_rows = read_jsonl(release / "estimation/analysis_panel.jsonl")
    saved_scales = read_json(release / "estimation/standardization_scales.json")
    maximum_beta_error = 0.0
    reconstructed_panel_rows = 0
    draw_cache: dict[str, list[dict[str, Any]]] = {}
    for backend_id in runner.BACKEND_IDS:
        semantic = runner.build_semantic_inputs(
            release,
            sample,
            futures_rows,
            backend_id,
        )
        saved_backend_panel = [
            row for row in panel_rows if row["backend_id"] == backend_id
        ]
        if canonical(semantic["panel_rows"]) != canonical(saved_backend_panel):
            raise ValidationError(f"reconstructed analysis panel mismatch: {backend_id}")
        if canonical(semantic["scales"]) != canonical(saved_scales[backend_id]):
            raise ValidationError(f"reconstructed standardization scales mismatch: {backend_id}")
        reconstructed_panel_rows += len(semantic["panel_rows"])
        replayed = independent_bootstrap_backend(semantic, panel_rows, replayed_plan)
        saved = read_jsonl(release / "estimation" / backend_id / "bootstrap_draws.jsonl")
        if len(saved) != len(replayed):
            raise ValidationError(f"bootstrap replay row-count mismatch: {backend_id}")
        for saved_row, replayed_row in zip(saved, replayed, strict=True):
            if (
                int(saved_row["draw_index"]) != int(replayed_row["draw_index"])
                or saved_row["sampled_calendar_years"] != replayed_row["sampled_calendar_years"]
                or saved_row["plan_sha256"] != replayed_row["plan_sha256"]
                or saved_row["schema_version"] != runner.SCHEMA + ":bootstrap-row"
                or saved_row["backend_id"] != backend_id
                or saved_row["specification"] != "basic_tadle_form"
                or saved_row["convention"] != runner.PRIMARY_CONVENTION
                or set(saved_row["betas"]) != {f"FF{horizon}" for horizon in runner.HORIZONS}
            ):
                raise ValidationError(f"bootstrap replay identity mismatch: {backend_id}")
            for label in (f"FF{horizon}" for horizon in runner.HORIZONS):
                if set(saved_row["betas"][label]) != set(runner.ARMS):
                    raise ValidationError(f"bootstrap arm-key mismatch: {backend_id}/{label}")
                for arm in runner.ARMS:
                    error = assert_finite_close(
                        f"bootstrap beta {backend_id}/{label}/{arm}",
                        saved_row["betas"][label][arm],
                        replayed_row["betas"][label][arm],
                    )
                    maximum_beta_error = max(maximum_beta_error, error)
        draw_cache[backend_id] = replayed
    return draw_cache, {
        "reconstructed_analysis_panel_rows": reconstructed_panel_rows,
        "reconstructed_standardization_scale_objects": len(runner.BACKEND_IDS),
        "reestimated_draws_per_backend": int(plan["public"]["draws"]),
        "reestimated_beta_values_per_backend": int(plan["public"]["draws"])
        * len(runner.HORIZONS)
        * len(runner.ARMS),
        "maximum_draw_beta_replay_error": maximum_beta_error,
    }


def verify_bootstrap_summaries(
    release: Path,
    plan: Mapping[str, Any],
    draw_cache: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    saved_contrasts = read_jsonl(release / "estimation/model_minus_official_contrasts.jsonl")
    saved_gains = read_jsonl(release / "estimation/distance_gains.jsonl")
    point_rows = read_jsonl(release / "estimation/coefficient_estimates.jsonl")
    point_lookup = {
        (row["backend_id"], row["horizon_label"], row["arm"]): float(row["beta_news_shock"])
        for row in point_rows
        if row["convention"] == runner.PRIMARY_CONVENTION
        and row["specification"] == "basic_tadle_form"
    }
    expected_contrast_keys = {
        (backend_id, f"FF{horizon}", arm)
        for backend_id in runner.BACKEND_IDS
        for horizon in runner.HORIZONS
        for arm in runner.GENERATED_ARMS
    }
    actual_contrast_keys = {
        (row["backend_id"], row["horizon_label"], row["arm"])
        for row in saved_contrasts
    }
    expected_gain_keys = {
        (backend_id, f"FF{horizon}", comparator)
        for backend_id in runner.BACKEND_IDS
        for horizon in runner.HORIZONS
        for comparator in ("chk1", "chk0")
    }
    actual_gain_keys = {
        (row["backend_id"], row["horizon_label"], row["comparator_arm"])
        for row in saved_gains
    }
    if (
        len(saved_contrasts) != len(actual_contrast_keys)
        or actual_contrast_keys != expected_contrast_keys
        or len(saved_gains) != len(actual_gain_keys)
        or actual_gain_keys != expected_gain_keys
    ):
        raise ValidationError("bootstrap summary Cartesian key coverage drift")
    maximum_error = 0.0
    for backend_id in runner.BACKEND_IDS:
        draws = draw_cache[backend_id]
        if len(draws) != int(plan["public"]["draws"]):
            raise ValidationError("bootstrap draw count mismatch")
        expected_years = tuple(plan["public"]["calendar_year_blocks"])
        for index, row in enumerate(draws):
            replayed = [expected_years[int(value)] for value in plan["year_indexes"][index]]
            if row["sampled_calendar_years"] != replayed or row["plan_sha256"] != plan["plan_sha256"]:
                raise ValidationError("saved bootstrap year draw mismatch")
    for backend_id in runner.BACKEND_IDS:
        backend_rows = [row for row in saved_contrasts if row["backend_id"] == backend_id]
        raw_ps = []
        for row in backend_rows:
            label = row["horizon_label"]
            arm = row["arm"]
            point_estimate = (
                point_lookup[(backend_id, label, arm)]
                - point_lookup[(backend_id, label, "official")]
            )
            values = np.asarray([
                draw["betas"][label][arm] - draw["betas"][label]["official"]
                for draw in draw_cache[backend_id]
            ])
            low, high = np.quantile(values, [0.025, 0.975], method="linear")
            raw = sign_p(values)
            raw_ps.append(raw)
            errors = (
                assert_finite_close("contrast estimate", row["estimate"], point_estimate),
                assert_finite_close("contrast estimate bp", row["estimate_basis_points"], 100.0 * point_estimate),
                assert_finite_close("contrast CI low", row["ci_95_low"], low),
                assert_finite_close("contrast CI high", row["ci_95_high"], high),
                assert_finite_close("contrast CI low bp", row["ci_95_low_basis_points"], 100.0 * low),
                assert_finite_close("contrast CI high bp", row["ci_95_high_basis_points"], 100.0 * high),
                assert_finite_close("contrast raw p", row["bootstrap_p_raw"], raw),
            )
            maximum_error = max(maximum_error, *errors)
            if (
                row["schema_version"] != runner.SCHEMA + ":model-minus-official-row"
                or row["convention"] != runner.PRIMARY_CONVENTION
                or row["specification"] != "basic_tadle_form"
                or int(row["bootstrap_draws"]) != int(plan["public"]["draws"])
                or row["paper_label"] != runner.PAPER_LABELS[arm]
                or row["family"] != "basic_beta_model_minus_official_across_3_models_x_4_horizons"
            ):
                raise ValidationError("bootstrap contrast metadata drift")
        for row, adjusted in zip(backend_rows, independent_holm(raw_ps), strict=True):
            maximum_error = max(
                maximum_error,
                assert_finite_close("contrast Holm p", row["holm_p"], adjusted),
            )

        backend_gains = [row for row in saved_gains if row["backend_id"] == backend_id]
        raw_ps = []
        for row in backend_gains:
            label = row["horizon_label"]
            comparator = row["comparator_arm"]
            point_estimate = (
                abs(
                    point_lookup[(backend_id, label, comparator)]
                    - point_lookup[(backend_id, label, "official")]
                )
                - abs(
                    point_lookup[(backend_id, label, "chk3")]
                    - point_lookup[(backend_id, label, "official")]
                )
            )
            values = np.asarray([
                abs(draw["betas"][label][comparator] - draw["betas"][label]["official"])
                - abs(draw["betas"][label]["chk3"] - draw["betas"][label]["official"])
                for draw in draw_cache[backend_id]
            ])
            low, high = np.quantile(values, [0.025, 0.975], method="linear")
            raw = sign_p(values)
            raw_ps.append(raw)
            errors = (
                assert_finite_close("gain estimate", row["estimate"], point_estimate),
                assert_finite_close("gain estimate bp", row["estimate_basis_points"], 100.0 * point_estimate),
                assert_finite_close("gain CI low", row["ci_95_low"], low),
                assert_finite_close("gain CI high", row["ci_95_high"], high),
                assert_finite_close("gain CI low bp", row["ci_95_low_basis_points"], 100.0 * low),
                assert_finite_close("gain CI high bp", row["ci_95_high_basis_points"], 100.0 * high),
                assert_finite_close("gain raw p", row["bootstrap_p_raw"], raw),
            )
            maximum_error = max(maximum_error, *errors)
            if (
                row["schema_version"] != runner.SCHEMA + ":distance-gain-row"
                or row["convention"] != runner.PRIMARY_CONVENTION
                or row["specification"] != "basic_tadle_form"
                or int(row["bootstrap_draws"]) != int(plan["public"]["draws"])
                or row["family"] != "chk2_distance_gains_across_2_comparators_x_4_horizons"
                or row["gain_definition"]
                != f"abs(beta_{comparator}-beta_official)-abs(beta_chk3-beta_official)"
            ):
                raise ValidationError("bootstrap distance-gain metadata drift")
        for row, adjusted in zip(backend_gains, independent_holm(raw_ps), strict=True):
            maximum_error = max(
                maximum_error,
                assert_finite_close("gain Holm p", row["holm_p"], adjusted),
            )
    return {
        "draws_per_backend": int(plan["public"]["draws"]),
        "contrast_rows": len(saved_contrasts),
        "distance_gain_rows": len(saved_gains),
        "maximum_summary_replay_error": maximum_error,
    }


def verify_manifest(release: Path) -> dict[str, Any]:
    manifest_path = release / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != EXPECTED_RELEASE_MANIFEST_SHA256:
        raise ValidationError(
            f"release manifest is not the preregistered sealed manifest: {manifest_hash}"
        )
    manifest = read_json(manifest_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("schema_version")
        != runner.SCHEMA + ":release-manifest"
        or manifest.get("exact_tadle_2022_replication") is not False
    ):
        raise ValidationError("release manifest status/schema/replication contract drift")
    checked = 0
    verified: dict[str, Any] = {}
    for name, binding in manifest["artifacts"].items():
        if name == "bootstrap_draws":
            for backend_id, nested in binding.items():
                verified[f"artifacts.bootstrap_draws.{backend_id}"] = verify_bound(
                    release / nested["path"], nested, allowed_root=release
                )
                checked += 1
        else:
            verified[f"artifacts.{name}"] = verify_bound(
                release / binding["path"], binding, allowed_root=release
            )
            checked += 1
    for name, binding in manifest["input_bindings"].items():
        if name == "official_minutes_scores":
            for backend_id, nested in binding.items():
                verified[f"input_bindings.official_minutes_scores.{backend_id}"] = verify_bound(
                    Path(nested["path"]), nested, allowed_root=ROOT
                )
                checked += 1
        else:
            verified[f"input_bindings.{name}"] = verify_bound(
                Path(binding["path"]), binding, allowed_root=ROOT
            )
            checked += 1

    for backend_id, embedded in manifest["statement_sentiment_manifests"].items():
        on_disk_path = release / "statements/sentiment" / backend_id / "manifest.json"
        on_disk = read_json(on_disk_path)
        if canonical(on_disk) != canonical(embedded):
            raise ValidationError(f"Statement sentiment manifest drift: {backend_id}")
        verified[f"statement_sentiment_manifests.{backend_id}.manifest"] = {
            "path": str(on_disk_path.resolve()),
            "bytes": on_disk_path.stat().st_size,
            "sha256": sha256_file(on_disk_path),
        }
        checked += 1
        for artifact_name, binding in embedded["artifacts"].items():
            verified[
                f"statement_sentiment_manifests.{backend_id}.artifacts.{artifact_name}"
            ] = verify_bound(
                release / binding["path"], binding, allowed_root=release
            )
            checked += 1
        model_binding = embedded["model_manifest"]
        verified[f"statement_sentiment_manifests.{backend_id}.model_manifest"] = verify_bound(
            Path(model_binding["path"]), model_binding, allowed_root=ROOT
        )
        checked += 1

    futures_manifest_path = release / "futures/manifest.json"
    futures_manifest = read_json(futures_manifest_path)
    if futures_manifest.get("schema_version") != runner.SCHEMA + ":futures-construction-manifest":
        raise ValidationError("futures construction manifest schema drift")
    verify_bound(
        release / futures_manifest["artifact"]["path"],
        futures_manifest["artifact"],
        allowed_root=release,
    )
    verified["supplemental.futures_manifest"] = {
        "path": str(futures_manifest_path.resolve()),
        "bytes": futures_manifest_path.stat().st_size,
        "sha256": sha256_file(futures_manifest_path),
    }
    result_summary_path = release / "result_summary.json"
    result_summary = read_json(result_summary_path)
    expected_summary = {
        "point_estimate_rows": 128,
        "primary_backend": runner.BACKEND_IDS[0],
        "primary_chk2_distance_gain_comparison_count": 8,
        "primary_chk2_distance_gain_holm_significant_count": 0,
        "primary_convention": runner.PRIMARY_CONVENTION,
        "primary_model_minus_official_comparison_count": 12,
        "primary_model_minus_official_holm_significant_count": 0,
        "smallest_primary_distance_gain_holm_p": 1.0,
        "smallest_primary_model_minus_official_holm_p": 1.0,
    }
    if result_summary != expected_summary:
        raise ValidationError("result summary content drift")
    verified["supplemental.result_summary"] = {
        "path": str(result_summary_path.resolve()),
        "bytes": result_summary_path.stat().st_size,
        "sha256": sha256_file(result_summary_path),
    }
    return {
        "sealed_release_manifest_sha256": manifest_hash,
        "release_artifact_and_input_bindings_checked": checked,
        "verified_bindings": verified,
    }


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"create-only validation output exists: {args.output_root}")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    validator_path = Path(__file__).resolve()
    validator_bytes = validator_path.read_bytes()
    validator_startup_sha256 = hashlib.sha256(validator_bytes).hexdigest()
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_root.name}.staging.",
            dir=args.output_root.parent,
        )
    )
    completed = False
    try:
        results = {
            "manifest": verify_manifest(args.release_root),
            "statements": verify_statement_replay(
                args.release_root,
                device=args.device,
                batch_size=args.batch_size,
            ),
            "futures": verify_futures(args.release_root),
            "point_estimates": verify_point_estimates(args.release_root),
        }
        plan = regenerate_plan(args.release_root)
        draw_cache, draw_replay = replay_bootstrap_draws(args.release_root, plan)
        results["bootstrap_draw_replay"] = draw_replay
        results["bootstrap"] = verify_bootstrap_summaries(args.release_root, plan, draw_cache)
        release_manifest_hash = sha256_file(args.release_root / "manifest.json")
        if release_manifest_hash != EXPECTED_RELEASE_MANIFEST_SHA256:
            raise ValidationError("release manifest changed during validation")
        if sha256_file(validator_path) != validator_startup_sha256:
            raise ValidationError("validator source changed during execution")
        report = {
            "schema_version": SCHEMA + ":report",
            "status": "passed",
            "release_root": str(args.release_root.resolve()),
            "release_manifest_sha256": release_manifest_hash,
            "validation_scope": results,
            "remaining_limitations": [
                "Statement model forward passes were replayed, but the separately sealed official/generated Minutes forward passes rely on the validated upstream benchmark.",
                "The semantic panel is reconstructed with the frozen producer contract and compared row-for-row with the sealed panel; independent NumPy implementations then validate every point estimate, all bootstrap coefficients, and all contrast summaries.",
                "The validator confirms the declared contract, not equivalence to Tadle's unavailable proprietary continuation series.",
            ],
        }
        write_new_json(staging / "validation_report.json", report)
        write_new_bytes(staging / "validator_source.py", validator_bytes)
        write_new_json(staging / "manifest.json", {
            "schema_version": SCHEMA + ":manifest",
            "status": "complete",
            "validated_release": str(args.release_root.resolve()),
            "validated_release_manifest_sha256": report["release_manifest_sha256"],
            "validator": {
                "live_path": str(validator_path),
                "startup_sha256": validator_startup_sha256,
                "preserved_path": "validator_source.py",
                "preserved_bytes": (staging / "validator_source.py").stat().st_size,
                "preserved_sha256": sha256_file(staging / "validator_source.py"),
            },
            "artifact": {
                "path": "validation_report.json",
                "bytes": (staging / "validation_report.json").stat().st_size,
                "sha256": sha256_file(staging / "validation_report.json"),
            },
        })
        os.replace(staging, args.output_root)
        completed = True
    finally:
        if not completed and staging.exists():
            shutil.rmtree(staging)
    print(canonical({"status": "passed", "output_root": str(args.output_root.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
