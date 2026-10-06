"""Run the single-view official-Minutes Treasury coefficient benchmark.

The benchmark is intentionally separate from the synthetic-reference artifacts.
It downloads and seals the official full Minutes pages, scores the official and
generated full documents with identical long-text classifiers, and compares
reduced-form DGS2 coefficients.  Generated text was not released historically;
no estimate from this module is a market-impact or causal-effect estimate.
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
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from jobs.eval import chk3_beta_statistics as statistics


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = Path(__file__).resolve()
RUN_ROOT = ROOT / "output/evaluation/main/chk3_beta_core8_merged_n2048_vllm_k5_20260816_v1"
DEFAULT_OUTPUT = ROOT / "output/evaluation/main/official_minutes_full_document_treasury_beta_1993_2025_v1"
PRE_ROSTER = ROOT / "dataset/processed/retrain_v2/chk3_minutes_external_holdout_1993_2008_all_regular_v1/official_meeting_roster.jsonl"
POST_ROSTER = ROOT / "dataset/processed/retrain_v2/chk3_minutes_post2008_2009_2025_fixed_core8_d1_v1/official_meeting_roster.jsonl"
GENERATED_DOCUMENTS = RUN_ROOT / "meeting_documents_n3840_v4_stochastic_schedule_v2/meeting_documents.v2.jsonl"
GENERATED_MANIFEST = RUN_ROOT / "meeting_documents_n3840_v4_stochastic_schedule_v2/manifest.json"
MARKET_PANEL = RUN_ROOT / "market_panel_fed_minutes_release_daily_v1/market_panel.jsonl"
MARKET_EXCLUSIONS = RUN_ROOT / "market_panel_fed_minutes_release_daily_v1/market_panel.exclusions.v1.jsonl"
RELEASE_LEDGER = RUN_ROOT / "market_panel_fed_minutes_release_daily_v1/official_minutes_release_dates.v1.jsonl"
MODEL_ROOT = ROOT / "models/evaluation/sentiment_core8_stochastic_schedule_v1"

DISTIL_ID = "distilbert_fomc_9c061b4_v1"
FINBERT_ID = "prosus_finbert_4556d13_v1"
BACKENDS = {
    DISTIL_ID: MODEL_ROOT / "achen0525--DistilBERT_FOMC_Classifier--9c061b4ce901418ff7839e9d89ca07156125d201",
    FINBERT_ID: MODEL_ROOT / "ProsusAI--finbert--4556d13015211d73dccd3fdd39d39232506f3e43",
}
PRIMARY_BACKEND = DISTIL_ID
CORE8 = (
    "Consumer-Price-Index-(CPI)",
    "GDP-Growth",
    "Government-Purchases",
    "Housing-Starts",
    "Industrial-Production",
    "Labour-Market",
    "Money-Supply",
    "Unemployment-Rate",
)
ARM_ORDER = ("official", "chk0", "chk1", "chk3")
SYNTHETIC_ARMS = ("chk0", "chk1", "chk3")
PAPER_LABELS = {
    "official": "Official FOMC Minutes",
    "chk0": "Model chk-0",
    "chk1": "Model chk-1 cp200",
    "chk3": "Model chk-2 cp318",
}
CONTRAST_LABELS = {
    "chk0_minus_official": "Model chk-0 minus official Minutes",
    "chk1_minus_official": "Model chk-1 cp200 minus official Minutes",
    "chk3_minus_official": "Model chk-2 cp318 minus official Minutes",
}
DISTANCE_LABELS = {
    "chk2_vs_chk1": "Distance gain: Model chk-2 over Model chk-1",
    "chk2_vs_chk0": "Distance gain: Model chk-2 over Model chk-0",
}
CONTENT_TOKENS = 510
OVERLAP_TOKENS = 128
WINDOW_STEP = CONTENT_TOKENS - OVERLAP_TOKENS
EXPECTED_MEETINGS = 256
EXPECTED_GENERATED = 3 * 256 * 5
EXPECTED_MARKET_ROWS = 255
EXPECTED_REGRESSION_N = 254
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_821
HAC_LAG = 4
SCHEMA = "official-minutes-full-document-treasury-beta-v1"


class BenchmarkError(RuntimeError):
    """A frozen-input, numerical, or publication contract failed closed."""


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


def file_binding(path: Path, *, relative_to: Path | None = None, rows: int | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise BenchmarkError(f"bound file missing or symlinked: {resolved}")
    shown = resolved if relative_to is None else resolved.relative_to(relative_to.resolve())
    result: dict[str, Any] = {"path": str(shown), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}
    if rows is not None:
        result["rows"] = rows
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise BenchmarkError(f"non-object JSONL row: {path}")
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


def runtime_contract() -> dict[str, Any]:
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
        raise BenchmarkError(f"runtime environment drift: {observed}")
    if platform.python_version() != "3.11.5" or np.__version__ != "1.24.3":
        raise BenchmarkError(
            f"runtime must be Python 3.11.5 / NumPy 1.24.3; observed {platform.python_version()} / {np.__version__}"
        )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "transformers": transformers.__version__,
        "environment": observed,
        "platform": platform.platform(),
    }


def load_roster() -> list[dict[str, Any]]:
    rows = read_jsonl(PRE_ROSTER) + read_jsonl(POST_ROSTER)
    rows.sort(key=lambda row: str(row["meeting_end_date"]))
    if len(rows) != EXPECTED_MEETINGS:
        raise BenchmarkError(f"official roster is N={len(rows)}, expected 256")
    for field in ("meeting_id", "meeting_end_date", "official_minutes_url"):
        values = [str(row[field]) for row in rows]
        if len(set(values)) != EXPECTED_MEETINGS:
            raise BenchmarkError(f"official roster {field} is not unique and complete")
    return rows


CHROME_SELECTORS = (
    "script", "style", "noscript", "nav", "footer", "header", "form", "button", "svg", "img",
    "#lastUpdate", ".lastUpdate", "#FooterLinks", "#secondaryFooterLinks", "#footer", ".footer",
    ".breadcrumb", "#t4_nav", ".shareDL", ".page-header", "#back-top", ".icon__backTop",
    "[role='navigation']", "[aria-label='breadcrumb']", "body > font[size='-1']",
)
CHROME_EXACT_TEXT = {
    "Return to top", "[Return to top]", "Back to Top", "FOMC", "Home | Monetary policy", "Home | Monetary Policy", "Print"
}


def clean_official_html(raw: bytes) -> tuple[str, dict[str, Any]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "html.parser")
    selected = None
    selector = None
    for candidate in ("#article", "#printThis", "#content", "body"):
        node = soup.select_one(candidate)
        if node is not None and len(node.get_text(" ", strip=True)) >= 1_000:
            selected, selector = node, candidate
            break
    if selected is None or selector is None:
        raise BenchmarkError("no supported official Minutes body container")
    fragment = BeautifulSoup(str(selected), "html.parser")
    removed_nodes = 0
    for chrome_selector in CHROME_SELECTORS:
        for node in fragment.select(chrome_selector):
            node.decompose()
            removed_nodes += 1
    for node in list(fragment.find_all(["a", "p", "div"])):
        if node.get_text(" ", strip=True) in CHROME_EXACT_TEXT:
            node.decompose()
            removed_nodes += 1
    text = fragment.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 1_000 or "Federal Open Market Committee" not in text:
        raise BenchmarkError(f"cleaned official body failed content gate: selector={selector}, chars={len(text)}")
    return text, {
        "selector": selector,
        "removed_nodes": removed_nodes,
        "characters": len(text),
        "whitespace_words": len(text.split()),
        "text_sha256": sha256_text(text),
        "no_silent_truncation": True,
    }


def fetch_and_prepare_official(staging: Path, roster: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(total=5, connect=5, read=5, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "FOMC academic reproducibility benchmark/1.0 (official Minutes archival fetch)"})
    source_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    for index, roster_row in enumerate(roster):
        url = str(roster_row["official_minutes_url"])
        response = session.get(url, timeout=(15, 90), allow_redirects=True)
        response.raise_for_status()
        raw = response.content
        if not raw:
            raise BenchmarkError(f"empty official HTTP body: {url}")
        meeting_id = str(roster_row["meeting_id"])
        end_date = str(roster_row["meeting_end_date"])
        expected_compact_dates = {
            str(roster_row["meeting_start_date"]).replace("-", ""),
            end_date.replace("-", ""),
        }
        if not any(value in response.url.lower() for value in expected_compact_dates):
            raise BenchmarkError(f"final URL does not bind meeting identity: {meeting_id}:{response.url}")
        raw_rel = Path("source/raw") / f"{meeting_id}.html"
        write_new_bytes(staging / raw_rel, raw)
        clean_text, clean_meta = clean_official_html(raw)
        source_rows.append({
            "schema_version": SCHEMA + ":official-source-row",
            "roster_index": index,
            "meeting_id": meeting_id,
            "meeting_start_date": roster_row["meeting_start_date"],
            "meeting_end_date": end_date,
            "requested_url": url,
            "final_url": response.url,
            "http_status": response.status_code,
            "content_type": response.headers.get("Content-Type"),
            "raw_path": str(raw_rel),
            "raw_bytes": len(raw),
            "raw_sha256": sha256_bytes(raw),
            "identity_bound_by_unique_roster_row_and_url_meeting_date": True,
        })
        document_rows.append({
            "schema_version": SCHEMA + ":official-document-row",
            "document_id": "official::" + end_date,
            "arm": "official",
            "paper_label": PAPER_LABELS["official"],
            "meeting_id": meeting_id,
            "generation_meeting_id": end_date,
            "meeting_start_date": roster_row["meeting_start_date"],
            "meeting_end_date": end_date,
            "replicate_id": None,
            "deterministic": True,
            "source_raw_path": str(raw_rel),
            "source_raw_sha256": sha256_bytes(raw),
            "document_text": clean_text,
            "document_text_sha256": clean_meta["text_sha256"],
            "cleaning": clean_meta,
        })
    if len(source_rows) != EXPECTED_MEETINGS or len(document_rows) != EXPECTED_MEETINGS:
        raise BenchmarkError("official source/document coverage drift")
    return source_rows, document_rows


def prepare_generated_documents() -> list[dict[str, Any]]:
    rows = read_jsonl(GENERATED_DOCUMENTS)
    if len(rows) != EXPECTED_GENERATED:
        raise BenchmarkError(f"generated document ledger is N={len(rows)}, expected 3840")
    keys: set[tuple[str, str, int]] = set()
    output: list[dict[str, Any]] = []
    for source in rows:
        arm = str(source["model_id"])
        meeting = str(source["meeting_id"])
        replicate = int(source["replicate_id"])
        key = (arm, meeting, replicate)
        if arm not in SYNTHETIC_ARMS or key in keys:
            raise BenchmarkError(f"invalid or duplicate generated document: {key}")
        keys.add(key)
        sections = sorted(source["sections"], key=lambda row: int(row["topic_order"]))
        topics = tuple(str(row["topic"]) for row in sections)
        rebuilt = "\n\n".join(str(row["answer"]) for row in sections)
        text = str(source["document_text"])
        if topics != CORE8 or len(sections) != 8 or rebuilt != text or sha256_text(text) != source["document_text_sha256"]:
            raise BenchmarkError(f"generated Core8 assembly contract drift: {key}")
        output.append({
            "schema_version": SCHEMA + ":generated-document-row",
            "document_id": source["document_id"],
            "arm": arm,
            "paper_label": PAPER_LABELS[arm],
            "meeting_id": meeting,
            "generation_meeting_id": meeting,
            "meeting_start_date": source["meeting_start_date"],
            "meeting_end_date": source["meeting_end_date"],
            "replicate_id": replicate,
            "replicate_seed": source["replicate_seed"],
            "deterministic": False,
            "topic_order": list(CORE8),
            "section_count": 8,
            "assembly_separator": "two_newlines_exact_section_answers",
            "document_text": text,
            "document_text_sha256": source["document_text_sha256"],
            "source_document_id": source["document_id"],
        })
    counts = Counter(row["arm"] for row in output)
    if counts != Counter({arm: 1_280 for arm in SYNTHETIC_ARMS}):
        raise BenchmarkError(f"generated arm counts drift: {counts}")
    meetings = {row["meeting_id"] for row in output}
    if len(meetings) != 256 or any(
        {row["replicate_id"] for row in output if row["arm"] == arm and row["meeting_id"] == meeting} != set(range(5))
        for arm in SYNTHETIC_ARMS for meeting in meetings
    ):
        raise BenchmarkError("generated meeting/replicate coverage drift")
    output.sort(key=lambda row: (ARM_ORDER.index(str(row["arm"])), str(row["meeting_id"]), int(row["replicate_id"])))
    return output


def prepare_inputs(staging: Path) -> dict[str, Any]:
    roster = load_roster()
    release_rows = read_jsonl(RELEASE_LEDGER)
    if len(release_rows) != 256 or [row["meeting_id"] for row in release_rows] != [row["meeting_id"] for row in roster]:
        raise BenchmarkError("release ledger and official roster do not match exactly")
    source_rows, official_rows = fetch_and_prepare_official(staging, roster)
    generated_rows = prepare_generated_documents()
    source_path = staging / "source/source_manifest.jsonl"
    official_path = staging / "documents/official_documents.jsonl"
    generated_path = staging / "documents/generated_documents.jsonl"
    write_new_jsonl(source_path, source_rows)
    write_new_jsonl(official_path, official_rows)
    write_new_jsonl(generated_path, generated_rows)
    selector_counts = Counter(row["cleaning"]["selector"] for row in official_rows)
    manifest = {
        "schema_version": SCHEMA + ":input-manifest",
        "created_at_utc": utc_now(),
        "view": "full_document_only",
        "core8_aligned_view_created": False,
        "official_sources": file_binding(source_path, relative_to=staging, rows=len(source_rows)),
        "official_documents": file_binding(official_path, relative_to=staging, rows=len(official_rows)),
        "generated_documents": file_binding(generated_path, relative_to=staging, rows=len(generated_rows)),
        "source_bindings": {
            "pre_roster": file_binding(PRE_ROSTER), "post_roster": file_binding(POST_ROSTER),
            "release_ledger": file_binding(RELEASE_LEDGER, rows=len(release_rows)),
            "generated_source": file_binding(GENERATED_DOCUMENTS, rows=len(generated_rows)),
            "generated_manifest": file_binding(GENERATED_MANIFEST),
            "market_panel": file_binding(MARKET_PANEL), "market_exclusions": file_binding(MARKET_EXCLUSIONS),
        },
        "coverage": {
            "official_pages": len(source_rows), "official_documents": len(official_rows),
            "generated_documents": len(generated_rows), "meetings": 256, "replicates_per_model_meeting": 5,
            "official_empty_bodies": sum(not row["document_text"] for row in official_rows),
            "cleaning_selector_counts": dict(sorted(selector_counts.items())),
            "silent_truncations": 0,
        },
        "construction_contract": {
            "official": "full official Minutes body after website chrome removal",
            "generated": "eight generated Core8 paragraphs joined by two newlines in frozen topic order",
            "topic_order": list(CORE8),
        },
    }
    write_new_json(staging / "documents/manifest.json", manifest)
    return manifest


def window_specs(tokenizer: Any, text: str) -> list[dict[str, Any]]:
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if not ids:
        raise BenchmarkError("empty tokenization")
    starts = [0]
    while starts[-1] + CONTENT_TOKENS < len(ids):
        starts.append(starts[-1] + WINDOW_STEP)
    spans = [(start, min(start + CONTENT_TOKENS, len(ids))) for start in starts]
    coverage = np.zeros(len(ids), dtype=np.int16)
    for start, end in spans:
        coverage[start:end] += 1
    if np.any(coverage <= 0):
        raise BenchmarkError("long-text windowing left uncovered tokens")
    result = []
    for index, (start, end) in enumerate(spans):
        content = ids[start:end]
        weight = float(np.sum(1.0 / coverage[start:end]))
        result.append({
            "window_index": index, "token_start": start, "token_end": end,
            "content_ids": content, "input_ids": [int(tokenizer.cls_token_id), *content, int(tokenizer.sep_token_id)],
            "aggregation_weight": weight, "content_token_ids_sha256": sha256_text(canonical(content)),
        })
    if not math.isclose(sum(row["aggregation_weight"] for row in result), len(ids), abs_tol=1e-8):
        raise BenchmarkError("coverage-corrected weights do not equal full token count")
    return result


def score_backend(staging: Path, backend_id: str, *, device: str, batch_size: int) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = BACKENDS[backend_id]
    model_manifest_path = model_dir / "model_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    signed = model_manifest["signed_score"]
    official_rows = read_jsonl(staging / "documents/official_documents.jsonl")
    generated_rows = read_jsonl(staging / "documents/generated_documents.jsonl")
    documents = official_rows + generated_rows
    output_dir = staging / "sentiment" / backend_id
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    if tokenizer.cls_token_id != 101 or tokenizer.sep_token_id != 102 or tokenizer.num_special_tokens_to_add(pair=False) != 2:
        raise BenchmarkError(f"special-token contract drift: {backend_id}")
    torch.manual_seed(0)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise BenchmarkError("CUDA requested but unavailable")
        torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval().to(device)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    document_meta: dict[str, dict[str, Any]] = {}
    accumulators: dict[str, dict[str, Any]] = defaultdict(lambda: {"weight": 0.0, "probs": [0.0, 0.0, 0.0]})
    window_path = output_dir / "window_scores.jsonl"
    window_count = 0
    pending: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    window_handle = window_path.open("x", encoding="utf-8", newline="\n")

    def flush() -> None:
        nonlocal window_count, pending
        if not pending:
            return
        padded = tokenizer.pad({"input_ids": [row[1]["input_ids"] for row in pending]}, padding=True, return_tensors="pt")
        padded = {key: value.to(device) for key, value in padded.items()}
        with torch.inference_mode():
            logits = model(**padded).logits.float()
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        for (document, spec), probs in zip(pending, probabilities, strict=True):
            if len(probs) != 3 or not math.isclose(sum(probs), 1.0, abs_tol=2e-6):
                raise BenchmarkError("classifier probability contract drift")
            doc_id = str(document["document_id"])
            weight = float(spec["aggregation_weight"])
            accumulators[doc_id]["weight"] += weight
            for i, probability in enumerate(probs):
                accumulators[doc_id]["probs"][i] += weight * float(probability)
            score = float(probs[int(signed["positive_index"])]) - float(probs[int(signed["negative_index"])])
            window_handle.write(canonical({
                "schema_version": SCHEMA + ":window-score-row", "backend_id": backend_id,
                "document_id": doc_id, "arm": document["arm"], "meeting_id": document["generation_meeting_id"],
                "replicate_id": document.get("replicate_id"), "window_index": spec["window_index"],
                "token_start": spec["token_start"], "token_end": spec["token_end"],
                "content_token_count": spec["token_end"] - spec["token_start"],
                "content_token_ids_sha256": spec["content_token_ids_sha256"],
                "aggregation_weight": weight, "probabilities": [float(value) for value in probs], "score": score,
            }) + "\n")
            window_count += 1
        pending = []

    try:
        for document in documents:
            specs = window_specs(tokenizer, str(document["document_text"]))
            document_meta[str(document["document_id"])] = {
                "window_count": len(specs), "content_token_count": specs[-1]["token_end"],
                "coverage_corrected_weight": sum(row["aggregation_weight"] for row in specs),
            }
            for spec in specs:
                pending.append((document, spec))
                if len(pending) >= batch_size:
                    flush()
        flush()
    finally:
        window_handle.close()

    score_rows: list[dict[str, Any]] = []
    for document in documents:
        doc_id = str(document["document_id"])
        meta = document_meta[doc_id]
        accumulator = accumulators[doc_id]
        if not math.isclose(accumulator["weight"], meta["content_token_count"], abs_tol=1e-7):
            raise BenchmarkError(f"document aggregation weight drift: {doc_id}")
        probs = [float(value) / accumulator["weight"] for value in accumulator["probs"]]
        score = probs[int(signed["positive_index"])] - probs[int(signed["negative_index"])]
        score_rows.append({
            "schema_version": SCHEMA + ":document-score-row", "backend_id": backend_id,
            "construct": model_manifest["construct"], "document_id": doc_id, "arm": document["arm"],
            "paper_label": document["paper_label"], "meeting_id": document["generation_meeting_id"],
            "meeting_end_date": document["meeting_end_date"], "replicate_id": document.get("replicate_id"),
            "document_text_sha256": document["document_text_sha256"], **meta,
            "label_order": model_manifest["label_order"], "probabilities": probs, "score": float(score),
            "no_silent_truncation": True,
        })
    score_path = output_dir / "document_scores.jsonl"
    write_new_jsonl(score_path, score_rows)
    cuda_properties = torch.cuda.get_device_properties(device) if device.startswith("cuda") else None
    manifest = {
        "schema_version": SCHEMA + ":sentiment-manifest", "created_at_utc": utc_now(),
        "backend_id": backend_id, "construct": model_manifest["construct"],
        "model_manifest": file_binding(model_manifest_path), "model_files": model_manifest["files"],
        "window_contract": {
            "content_tokens": CONTENT_TOKENS, "overlap_tokens": OVERLAP_TOKENS, "step": WINDOW_STEP,
            "sequence": "[CLS] + content + [SEP]", "aggregation": "coverage-corrected probability mean",
            "signed_score": signed,
        },
        "coverage": {
            "documents": len(score_rows), "official_documents": len(official_rows), "generated_documents": len(generated_rows),
            "windows": window_count, "empty_documents": 0, "silent_truncations": 0,
        },
        "artifacts": {
            "window_scores": file_binding(window_path, relative_to=staging, rows=window_count),
            "document_scores": file_binding(score_path, relative_to=staging, rows=len(score_rows)),
        },
        "runtime": {
            **runtime_contract(), "torch": torch.__version__, "device": device, "batch_size": batch_size,
            "cuda_device_name": None if cuda_properties is None else cuda_properties.name,
            "cuda_total_memory_bytes": None if cuda_properties is None else cuda_properties.total_memory,
            "deterministic_algorithms": True, "tf32": False,
        },
    }
    write_new_json(output_dir / "manifest.json", manifest)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return manifest


def design_matrix(current: np.ndarray, lag: np.ndarray, vix: np.ndarray, months: np.ndarray) -> np.ndarray:
    return np.column_stack((
        np.ones(len(current), dtype=np.float64), current, lag, vix,
        np.sin(2.0 * np.pi * months / 12.0), np.cos(2.0 * np.pi * months / 12.0),
    ))


def strict_beta(y: np.ndarray, matrix: np.ndarray) -> float:
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
    if int(rank) != matrix.shape[1]:
        raise BenchmarkError("rank-deficient bootstrap design")
    return float(coefficients[1])


def prepare_estimation_inputs(staging: Path, backend_id: str) -> dict[str, Any]:
    scores = read_jsonl(staging / "sentiment" / backend_id / "document_scores.jsonl")
    official = {str(row["meeting_id"]): float(row["score"]) for row in scores if row["arm"] == "official"}
    synthetic: dict[str, dict[str, dict[int, float]]] = {arm: defaultdict(dict) for arm in SYNTHETIC_ARMS}
    for row in scores:
        arm = str(row["arm"])
        if arm in SYNTHETIC_ARMS:
            synthetic[arm][str(row["meeting_id"])][int(row["replicate_id"])] = float(row["score"])
    if len(official) != 256 or any(len(synthetic[arm]) != 256 for arm in SYNTHETIC_ARMS):
        raise BenchmarkError("sentiment meeting coverage drift")
    if any(set(reps) != set(range(5)) for arm in SYNTHETIC_ARMS for reps in synthetic[arm].values()):
        raise BenchmarkError("sentiment replicate coverage drift")
    release_rows = sorted(read_jsonl(RELEASE_LEDGER), key=lambda row: str(row["meeting_end_date"]))
    chronology = [str(row["generation_meeting_id"]) for row in release_rows]
    previous = {current: prior for prior, current in zip(chronology[:-1], chronology[1:], strict=True)}
    market_rows = sorted(read_jsonl(MARKET_PANEL), key=lambda row: str(row["release_date"]))
    if len(market_rows) != EXPECTED_MARKET_ROWS:
        raise BenchmarkError("market panel row count drift")
    pairs = [(previous[str(row["generation_meeting_id"])], row) for row in market_rows if str(row["generation_meeting_id"]) in previous]
    if len(pairs) != EXPECTED_REGRESSION_N:
        raise BenchmarkError(f"full regression N drift: {len(pairs)}")
    pre_ids = [str(row["generation_meeting_id"]) for row in release_rows if str(row["meeting_end_date"]) <= "2008-12-31"]
    if len(pre_ids) != 128:
        raise BenchmarkError("official pre-2009 scale does not contain 128 meetings")
    scale_values = np.asarray([official[mid] for mid in pre_ids], dtype=np.float64)
    mu, sd = float(scale_values.mean()), float(scale_values.std(ddof=1))
    if not math.isfinite(sd) or sd <= 0:
        raise BenchmarkError("official pre-2009 scale standard deviation is invalid")
    union_ids = tuple(dict.fromkeys([prior for prior, _ in pairs] + [str(row["generation_meeting_id"]) for _, row in pairs]))
    union_index = {mid: index for index, mid in enumerate(union_ids)}
    current_ids = tuple(str(row["generation_meeting_id"]) for _, row in pairs)
    lag_ids = tuple(prior for prior, _ in pairs)
    return {
        "official": official, "synthetic": synthetic, "release_rows": release_rows, "market_rows": market_rows,
        "pairs": pairs, "pre_ids": pre_ids, "mu": mu, "sd": sd, "union_ids": union_ids, "union_index": union_index,
        "current_ids": current_ids, "lag_ids": lag_ids,
        "current_indexes": np.asarray([union_index[mid] for mid in current_ids], dtype=np.int64),
        "lag_indexes": np.asarray([union_index[mid] for mid in lag_ids], dtype=np.int64),
        "y": np.asarray([float(row["dgs2_release_minus_previous_bp"]) for _, row in pairs], dtype=np.float64),
        "vix": np.asarray([float(row["vix_release_minus_previous_log_pct"]) for _, row in pairs], dtype=np.float64),
        "months": np.asarray([int(str(row["release_date"])[5:7]) for _, row in pairs], dtype=np.float64),
        "design_blocks": tuple(str(row["release_date"])[:4] for _, row in pairs),
        "release_year": {str(row["generation_meeting_id"]): str(row["release_date"])[:4] for row in release_rows},
    }


def contrast_values(betas: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    contrasts = {f"{arm}_minus_official": float(betas[arm] - betas["official"]) for arm in SYNTHETIC_ARMS}
    distances = {arm: float(abs(betas[arm] - betas["official"])) for arm in SYNTHETIC_ARMS}
    gains = {
        "chk2_vs_chk1": float(distances["chk1"] - distances["chk3"]),
        "chk2_vs_chk0": float(distances["chk0"] - distances["chk3"]),
    }
    return contrasts, distances, gains


def estimate_backend(staging: Path, backend_id: str, *, write: bool) -> dict[str, Any]:
    data = prepare_estimation_inputs(staging, backend_id)
    mu, sd = data["mu"], data["sd"]
    union_ids = data["union_ids"]
    official_union = np.asarray([(data["official"][mid] - mu) / sd for mid in union_ids], dtype=np.float64)
    synthetic_matrices = {
        arm: np.asarray([[(data["synthetic"][arm][mid][rep] - mu) / sd for rep in range(5)] for mid in union_ids], dtype=np.float64)
        for arm in SYNTHETIC_ARMS
    }
    point_vectors = {"official": official_union, **{arm: values.mean(axis=1) for arm, values in synthetic_matrices.items()}}
    point_rows: list[dict[str, Any]] = []
    point_betas: dict[str, float] = {}
    for arm in ARM_ORDER:
        vector = point_vectors[arm]
        matrix = design_matrix(vector[data["current_indexes"]], vector[data["lag_indexes"]], data["vix"], data["months"])
        result = statistics.ols_hac(data["y"], matrix, hac_lag=HAC_LAG)
        point_betas[arm] = float(result.coefficients[1])
        point_rows.append({
            "schema_version": SCHEMA + ":coefficient-row", "backend_id": backend_id, "arm": arm,
            "paper_label": PAPER_LABELS[arm], "nobs": result.nobs, "beta_current": float(result.coefficients[1]),
            "hac_se_current": float(result.standard_errors[1]), "beta_lag": float(result.coefficients[2]),
            "hac_se_lag": float(result.standard_errors[2]), "r_squared": float(result.r_squared),
            "adjusted_r_squared": float(result.adjusted_r_squared), "hac_lag": HAC_LAG,
        })
    point_contrasts, point_distances, point_gains = contrast_values(point_betas)

    block_ids = tuple(dict.fromkeys(data["design_blocks"]))
    plan = statistics.make_paired_block_bootstrap_plan(
        block_ids=block_ids, meeting_ids=union_ids,
        meeting_block_ids=[data["release_year"][mid] for mid in union_ids],
        replicate_ids_by_meeting={mid: tuple(range(5)) for mid in union_ids},
        draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, replicates_per_draw=5,
    )
    rows_by_block = {
        block: np.asarray([i for i, value in enumerate(data["design_blocks"]) if value == block], dtype=np.int64)
        for block in block_ids
    }
    draw_rows: list[dict[str, Any]] = []
    for draw in range(BOOTSTRAP_DRAWS):
        means = statistics.resampled_meeting_means(synthetic_matrices, plan, draw_index=draw)
        vectors = {"official": official_union, **means}
        sampled_rows = np.concatenate([rows_by_block[block_ids[int(i)]] for i in plan.sampled_block_indices[draw]])
        betas = {}
        for arm in ARM_ORDER:
            vector = vectors[arm]
            matrix = design_matrix(vector[data["current_indexes"]], vector[data["lag_indexes"]], data["vix"], data["months"])
            betas[arm] = strict_beta(data["y"][sampled_rows], matrix[sampled_rows])
        contrasts, distances, gains = contrast_values(betas)
        draw_rows.append({
            "schema_version": SCHEMA + ":bootstrap-row", "backend_id": backend_id, "draw_index": draw,
            "plan_sha256": plan.sha256, "sampled_rows": int(len(sampled_rows)), "betas": betas,
            "model_minus_official": contrasts, "absolute_distances": distances, "distance_gains": gains,
        })

    contrast_rows: list[dict[str, Any]] = []
    raw_ps = []
    for contrast_id, point in point_contrasts.items():
        values = np.asarray([row["model_minus_official"][contrast_id] for row in draw_rows], dtype=np.float64)
        low, high = statistics.percentile_interval(values)
        p = statistics.two_sided_bootstrap_p_value(values)
        raw_ps.append(p)
        contrast_rows.append({
            "schema_version": SCHEMA + ":contrast-row", "backend_id": backend_id, "family": "three_model_minus_official",
            "contrast_id": contrast_id, "label": CONTRAST_LABELS[contrast_id], "estimate": point,
            "ci_95_low": low, "ci_95_high": high, "bootstrap_p_raw": p, "bootstrap_draws": BOOTSTRAP_DRAWS,
        })
    for row, adjusted in zip(contrast_rows, statistics.holm_adjust(raw_ps), strict=True):
        row["holm_p"] = adjusted

    gain_rows: list[dict[str, Any]] = []
    raw_gain_ps = []
    for gain_id, point in point_gains.items():
        values = np.asarray([row["distance_gains"][gain_id] for row in draw_rows], dtype=np.float64)
        low, high = statistics.percentile_interval(values)
        p = statistics.two_sided_bootstrap_p_value(values)
        raw_gain_ps.append(p)
        gain_rows.append({
            "schema_version": SCHEMA + ":distance-gain-row", "backend_id": backend_id, "family": "two_chk2_distance_gains",
            "gain_id": gain_id, "label": DISTANCE_LABELS[gain_id], "estimate": point,
            "ci_95_low": low, "ci_95_high": high, "bootstrap_p_raw": p, "bootstrap_draws": BOOTSTRAP_DRAWS,
        })
    for row, adjusted in zip(gain_rows, statistics.holm_adjust(raw_gain_ps), strict=True):
        row["holm_p"] = adjusted

    result = {
        "backend_id": backend_id, "scale": {"source": "128 official pre-2009 full documents", "n": 128, "mean": mu, "sd": sd, "ddof": 1},
        "point_rows": point_rows, "point_betas": point_betas, "point_contrasts": point_contrasts,
        "point_distances": point_distances, "point_gains": point_gains,
        "contrast_rows": contrast_rows, "gain_rows": gain_rows, "draw_rows": draw_rows,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "plan_sha256": plan.sha256, "calendar_year_blocks": list(block_ids)},
        "design": {
            "nobs": len(data["pairs"]), "outcome": "DGS2 release-date close minus previous-valid close (basis points)",
            "vix_control": "VIX release-date minus previous-valid log percent change", "hac_lag": HAC_LAG,
            "current_meeting_start": data["current_ids"][0], "current_meeting_end": data["current_ids"][-1],
            "first_roster_meeting_dropped_for_lag": "1993-02-03",
            "market_incomplete_meeting": "2004-09-21",
            "market_incomplete_meeting_retained_as_lag_for": "2004-11-10",
        },
    }
    if write:
        out = staging / "estimation" / backend_id
        out.mkdir(parents=True, exist_ok=False)
        write_new_jsonl(out / "coefficient_estimates.jsonl", point_rows)
        write_new_jsonl(out / "bootstrap_draws.jsonl", draw_rows)
        write_new_jsonl(out / "model_minus_official_contrasts.jsonl", contrast_rows)
        write_new_jsonl(out / "distance_gains.jsonl", gain_rows)
        manifest = {
            "schema_version": SCHEMA + ":estimation-manifest", "created_at_utc": utc_now(),
            "backend_id": backend_id, "scale": result["scale"], "design": result["design"], "bootstrap": result["bootstrap"],
            "artifacts": {
                "coefficient_estimates": file_binding(out / "coefficient_estimates.jsonl", relative_to=staging, rows=len(point_rows)),
                "bootstrap_draws": file_binding(out / "bootstrap_draws.jsonl", relative_to=staging, rows=len(draw_rows)),
                "model_minus_official_contrasts": file_binding(out / "model_minus_official_contrasts.jsonl", relative_to=staging, rows=len(contrast_rows)),
                "distance_gains": file_binding(out / "distance_gains.jsonl", relative_to=staging, rows=len(gain_rows)),
            },
        }
        write_new_json(out / "manifest.json", manifest)
    return result


def format_number(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def render_table(result: Mapping[str, Any]) -> str:
    lines = [
        f"### {result['backend_id']}", "",
        "| Document source | Current sentiment coefficient | HAC SE | Lagged sentiment coefficient | HAC SE | N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["point_rows"]:
        lines.append(
            f"| {row['paper_label']} | {format_number(row['beta_current'])} | {format_number(row['hac_se_current'])} | "
            f"{format_number(row['beta_lag'])} | {format_number(row['hac_se_lag'])} | {row['nobs']} |"
        )
    lines.extend(["", "| Model-minus-official contrast | Estimate | 95% paired bootstrap CI | Raw p | Holm p |", "|---|---:|---:|---:|---:|"])
    for row in result["contrast_rows"]:
        lines.append(
            f"| {row['label']} | {format_number(row['estimate'])} | [{format_number(row['ci_95_low'])}, {format_number(row['ci_95_high'])}] | "
            f"{format_number(row['bootstrap_p_raw'], 6)} | {format_number(row['holm_p'], 6)} |"
        )
    lines.extend(["", "| Absolute-distance comparison | Gain | 95% paired bootstrap CI | Raw p | Holm p |", "|---|---:|---:|---:|---:|"])
    for row in result["gain_rows"]:
        lines.append(
            f"| {row['label']} | {format_number(row['estimate'])} | [{format_number(row['ci_95_low'])}, {format_number(row['ci_95_high'])}] | "
            f"{format_number(row['bootstrap_p_raw'], 6)} | {format_number(row['holm_p'], 6)} |"
        )
    return "\n".join(lines)


def render_report(staging: Path, results: Mapping[str, Mapping[str, Any]], input_manifest: Mapping[str, Any]) -> None:
    sections = [render_table(results[backend]) for backend in (DISTIL_ID, FINBERT_ID)]
    selector_counts = input_manifest["coverage"]["cleaning_selector_counts"]
    report = f"""# Official-Minutes Full-Document Treasury Coefficient Benchmark

## Scope and estimand

This create-only benchmark compares historical reduced-form Treasury coefficients across four document sources: official FOMC Minutes, Model chk-0, Model chk-1 cp200, and Model chk-2 cp318 (repository arm `chk3`). It does not test textual similarity. Official Minutes were released historically; generated documents were not. Consequently, the generated-text coefficients cannot be interpreted as market impacts or causal effects. The comparison is a document-source coefficient diagnostic, and the official full Minutes contain a broader information set than the concatenated Core8 model outputs.

Only the full-document view is reported. No Core8 label parser, topic mapping, missing-topic zero assignment, or Core8-aligned official robustness view was created. Existing synthetic-reference artifacts were neither overwritten nor relabeled.

## Inputs and construction

The frozen population contains 256 regular FOMC meetings from 1993 through 2025. For each meeting, the official HTML was downloaded from the roster URL and sealed with its final URL, raw bytes, SHA-256 digest, and roster identity. Cleaning selected one official body container and removed website chrome only. Selector counts were `{json.dumps(selector_counts, sort_keys=True)}`. All 256 cleaned bodies are nonempty and were scored without truncation.

Each generated meeting document joins eight frozen outputs in the order CPI, GDP Growth, Government Purchases, Housing Starts, Industrial Production, Labour Market, Money Supply, and Unemployment Rate. There are five stochastic documents per model and meeting, or 3,840 generated documents in total. Official and generated documents use the same 510-content-token windows, 128-token overlap, coverage-corrected window weighting, and classifier-specific signed probability score.

For each backend, the mean and sample standard deviation are frozen from the 128 pre-2009 official full-document scores and then applied to every arm. Arms are never standardized separately. DistilBERT FOMC hawkish-minus-dovish stance is primary; ProsusAI FinBERT positive-minus-negative financial valence is a separately reported robustness construct and is not pooled with DistilBERT.

## Regression and paired inference

The outcome is the DGS2 release-date close minus the previous-valid close in basis points. Each arm is estimated separately with current standardized sentiment, the immediately preceding FOMC meeting's standardized sentiment, the VIX release-to-previous log change, and annual sine/cosine terms. Point-estimate covariance is Newey--West HAC with lag four.

The market panel has 255 complete rows. The first roster meeting has no sentiment lag and is excluded, producing the prespecified full-panel regression N=254. The 2004-09-21 meeting lacks a complete market window and is excluded as a current outcome, but its sentiment remains the immediate-meeting lag for 2004-11-10.

Uncertainty uses 10,000 paired calendar-year block bootstrap draws. All four document sources share each year-block draw; all three generated arms share each meeting-level five-replicate resample; official Minutes are not resampled within meeting. Confidence intervals are percentiles of coefficient differences computed within each draw. The three model-minus-official tests form one Holm family. The two Model-chk-2 absolute-distance gains form a separate Holm family, with distances recomputed in every draw.

## Results

{chr(10).join(sections)}

## Interpretation limits

A confidence interval that crosses zero means that this design did not detect a distinguishable coefficient difference; it is not evidence of statistical equivalence. A positive distance gain means Model chk-2's coefficient is closer in absolute point-estimate distance than the named comparator, but it does not show that either coefficient is equivalent to the official coefficient. Post-2008 documents overlap the broader model-development period, so the 1993--2025 panel is descriptive rather than a wholly external model evaluation.
"""
    write_new_text(staging / "technical_report.md", report)
    table = "# English Result Tables\n\n" + "\n\n".join(sections) + "\n"
    write_new_text(staging / "result_tables.md", table)


def validate_sentiment_aggregation(staging: Path, backend_id: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BACKENDS[backend_id], local_files_only=True, use_fast=True)
    documents = read_jsonl(staging / "documents/official_documents.jsonl") + read_jsonl(staging / "documents/generated_documents.jsonl")
    windows = read_jsonl(staging / "sentiment" / backend_id / "window_scores.jsonl")
    scores = {row["document_id"]: row for row in read_jsonl(staging / "sentiment" / backend_id / "document_scores.jsonl")}
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in windows:
        by_document[str(row["document_id"])].append(row)
    for document in documents:
        doc_id = str(document["document_id"])
        specs = window_specs(tokenizer, str(document["document_text"]))
        observed = sorted(by_document[doc_id], key=lambda row: int(row["window_index"]))
        if len(specs) != len(observed):
            raise BenchmarkError(f"deep window count replay failed: {backend_id}:{doc_id}")
        weighted = np.zeros(3, dtype=np.float64)
        total = 0.0
        for spec, row in zip(specs, observed, strict=True):
            replay = (spec["window_index"], spec["token_start"], spec["token_end"], spec["content_token_ids_sha256"])
            stored = (row["window_index"], row["token_start"], row["token_end"], row["content_token_ids_sha256"])
            if replay != stored or not math.isclose(spec["aggregation_weight"], float(row["aggregation_weight"]), abs_tol=1e-12):
                raise BenchmarkError(f"deep token/window replay failed: {backend_id}:{doc_id}")
            weighted += float(row["aggregation_weight"]) * np.asarray(row["probabilities"], dtype=np.float64)
            total += float(row["aggregation_weight"])
        probs = weighted / total
        if not np.allclose(probs, np.asarray(scores[doc_id]["probabilities"]), atol=1e-12, rtol=0.0):
            raise BenchmarkError(f"deep probability aggregation replay failed: {backend_id}:{doc_id}")
    return {"documents_retokenized": len(documents), "windows_replayed": len(windows), "aggregation_exact_tolerance": 1e-12}


def deep_validate(staging: Path, results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    source_rows = read_jsonl(staging / "source/source_manifest.jsonl")
    official_rows = {row["meeting_id"]: row for row in read_jsonl(staging / "documents/official_documents.jsonl")}
    for source in source_rows:
        raw_path = staging / str(source["raw_path"])
        raw = raw_path.read_bytes()
        if len(raw) != source["raw_bytes"] or sha256_bytes(raw) != source["raw_sha256"]:
            raise BenchmarkError(f"official source replay failed: {source['meeting_id']}")
        text, meta = clean_official_html(raw)
        document = official_rows[source["meeting_id"]]
        if text != document["document_text"] or meta != document["cleaning"]:
            raise BenchmarkError(f"official cleaning replay failed: {source['meeting_id']}")
    generated_replay = prepare_generated_documents()
    if canonical(generated_replay) != canonical(read_jsonl(staging / "documents/generated_documents.jsonl")):
        raise BenchmarkError("generated-document replay failed")
    sentiment_validation = {backend: validate_sentiment_aggregation(staging, backend) for backend in BACKENDS}
    estimation_validation = {}
    for backend in BACKENDS:
        replay = estimate_backend(staging, backend, write=False)
        stored_dir = staging / "estimation" / backend
        comparisons = {
            "coefficient_estimates": (replay["point_rows"], read_jsonl(stored_dir / "coefficient_estimates.jsonl")),
            "bootstrap_draws": (replay["draw_rows"], read_jsonl(stored_dir / "bootstrap_draws.jsonl")),
            "model_minus_official_contrasts": (replay["contrast_rows"], read_jsonl(stored_dir / "model_minus_official_contrasts.jsonl")),
            "distance_gains": (replay["gain_rows"], read_jsonl(stored_dir / "distance_gains.jsonl")),
        }
        for name, (rebuilt, stored) in comparisons.items():
            if canonical(rebuilt) != canonical(stored):
                raise BenchmarkError(f"deep estimation replay failed: {backend}:{name}")
        estimation_validation[backend] = {"nobs": replay["design"]["nobs"], "bootstrap_draws_replayed": len(replay["draw_rows"]), "exact_canonical_match": True}
    return {
        "schema_version": SCHEMA + ":validation-report", "validated_at_utc": utc_now(), "status": "passed",
        "official_sources_rehashed_and_recleaned": len(source_rows), "generated_documents_rebuilt": len(generated_replay),
        "sentiment": sentiment_validation, "estimation": estimation_validation,
        "join_gates": {"unique_complete_meetings": 256, "official_empty_bodies": 0, "generated_duplicate_keys": 0, "regression_n": 254},
    }


def run(output: Path, *, device: str, batch_size: int) -> None:
    contract = runtime_contract()
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise BenchmarkError(f"create-only output already exists: {output}")
    staging = output.parent / ("." + output.name + ".staging-" + next(tempfile._get_candidate_names()))
    if staging.exists():
        raise BenchmarkError(f"staging path unexpectedly exists: {staging}")
    staging.mkdir(parents=True)
    try:
        input_manifest = prepare_inputs(staging)
        sentiment_manifests = {backend: score_backend(staging, backend, device=device, batch_size=batch_size) for backend in BACKENDS}
        results = {backend: estimate_backend(staging, backend, write=True) for backend in BACKENDS}
        render_report(staging, results, input_manifest)
        validation = deep_validate(staging, results)
        write_new_json(staging / "validation_report.json", validation)
        final_manifest = {
            "schema_version": SCHEMA + ":manifest", "created_at_utc": utc_now(), "status": "complete",
            "evaluation_scope": "single full-document official-Minutes Treasury coefficient comparison",
            "primary_backend": PRIMARY_BACKEND, "robustness_backend": FINBERT_ID,
            "runtime": contract,
            "implementation": file_binding(IMPLEMENTATION),
            "inputs": file_binding(staging / "documents/manifest.json", relative_to=staging),
            "sentiment_manifests": {backend: file_binding(staging / "sentiment" / backend / "manifest.json", relative_to=staging) for backend in BACKENDS},
            "estimation_manifests": {backend: file_binding(staging / "estimation" / backend / "manifest.json", relative_to=staging) for backend in BACKENDS},
            "reports": {
                "technical_report": file_binding(staging / "technical_report.md", relative_to=staging),
                "result_tables": file_binding(staging / "result_tables.md", relative_to=staging),
                "validation": file_binding(staging / "validation_report.json", relative_to=staging),
            },
            "interpretation": {
                "estimand": "historical reduced-form document-source coefficient comparison",
                "not_semantic_similarity": True, "not_causal": True, "not_equivalence_test": True,
                "official_and_generated_information_scope_differs": True,
            },
            "existing_synthetic_reference_artifacts_modified": False,
        }
        write_new_json(staging / "manifest.json", final_manifest)
        staging.rename(output)
    except Exception:
        # Failed staging is retained for forensic inspection; the final create-only root is never partially published.
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--batch-size", type=int, default=32)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run(args.output, device=args.device, batch_size=args.batch_size)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
