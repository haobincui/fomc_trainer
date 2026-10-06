"""Versioned Core8 sentiment scoring for the sealed stochastic-schedule panel.

The primary backend is a frozen Lucca--Trebbi-inspired hawk/dove lexicon.
Two independently reported neural robustness backends are also supported:

* ``achen0525/DistilBERT_FOMC_Classifier`` for dovish/hawkish/neutral stance;
* ``ProsusAI/finbert`` for positive/negative/neutral financial valence.

The two neural constructs are never pooled.  Every backend emits lossless
window-, topic-, and meeting-level files.  The regression-facing
``meeting_scores.jsonl`` is deep-rebuilt from the topic rows by its loader.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jobs.eval import chk3_beta_core8_merged_contract as data_contract
from jobs.eval import prepare_chk3_beta_core8_merged_vllm_k5 as preparation
from jobs.eval import (
    seal_chk3_beta_core8_vllm_k5_dual_dp1_stochastic_schedule_v2_suite as generation_suite,
)
from open_r1.provenance import sha256_file, sha256_text
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_PATH = Path(__file__).resolve()
# Captured before any command executes.  A long-running scorer must never seal
# outputs against source bytes different from the code loaded by that process.
IMPLEMENTATION_SHA256_AT_IMPORT = sha256_file(IMPLEMENTATION_PATH)
RUN_ROOT = ROOT / (
    "output/evaluation/main/chk3_beta_core8_merged_n2048_vllm_k5_20260816_v1"
)
DEFAULT_COHORT = RUN_ROOT / "preparation/cohort_n2048_k5.v1.json"
DEFAULT_GENERATION_SUITE = RUN_ROOT / (
    "generation_formal_n2048_k5_three_models_v4_stochastic_schedule_v2/manifest.json"
)
DEFAULT_OUTPUT_ROOT = RUN_ROOT / "sentiment_core8_stochastic_schedule_v1"
DEFAULT_MODEL_ROOT = ROOT / "models/evaluation/sentiment_core8_stochastic_schedule_v1"

EVALUATION_ID = "chk3-beta-core8-sentiment-stochastic-schedule-v1"
INVENTORY_SCHEMA = "chk3-beta-core8-sentiment-text-inventory-row-v1"
INVENTORY_MANIFEST_SCHEMA = "chk3-beta-core8-sentiment-text-inventory-manifest-v1"
WINDOW_SCHEMA = "chk3-beta-core8-sentiment-window-score-row-v1"
TOPIC_SCHEMA = "chk3-beta-core8-sentiment-topic-score-row-v1"
MEETING_SCHEMA = "chk3-beta-core8-sentiment-meeting-score-row-v1"
BACKEND_MANIFEST_SCHEMA = "chk3-beta-core8-sentiment-backend-manifest-v1"
MODEL_MANIFEST_SCHEMA = "chk3-beta-core8-sentiment-model-manifest-v1"
SUITE_MANIFEST_SCHEMA = "chk3-beta-core8-sentiment-suite-manifest-v1"

ARM_ORDER = ("reference", "chk0", "chk1", "chk3")
GENERATION_MODEL_ORDER = ("chk0", "chk1", "chk3")
REPLICATE_COUNT = len(preparation.REPLICATE_SEEDS)
EXPECTED_REFERENCE_TOPICS = data_contract.EXPECTED_ROWS
EXPECTED_GENERATED_TOPICS_PER_ARM = data_contract.EXPECTED_ROWS * REPLICATE_COUNT
EXPECTED_TEXTS = EXPECTED_REFERENCE_TOPICS + len(GENERATION_MODEL_ORDER) * (
    EXPECTED_GENERATED_TOPICS_PER_ARM
)
EXPECTED_MEETING_ROWS = data_contract.EXPECTED_MEETINGS * (
    1 + len(GENERATION_MODEL_ORDER) * REPLICATE_COUNT
)
TOPIC_COUNT = len(data_contract.CORE_TOPICS)

CONTENT_TOKENS = 510
OVERLAP_TOKENS = 128
WINDOW_STEP = CONTENT_TOKENS - OVERLAP_TOKENS
FSYNC_EVERY_ROWS = 64

LEXICON_BACKEND = "lucca_trebbi_hawk_dove_lexicon_v1"
DISTIL_BACKEND = "distilbert_fomc_9c061b4_v1"
FINBERT_BACKEND = "prosus_finbert_4556d13_v1"
BACKEND_ORDER = (LEXICON_BACKEND, DISTIL_BACKEND, FINBERT_BACKEND)


@dataclass(frozen=True)
class ModelContract:
    backend_id: str
    repo_id: str
    revision: str
    directory_name: str
    weights_file: str
    weights_sha256: str
    readme_sha256: str
    config_sha256: str
    labels: tuple[str, str, str]
    positive_index: int
    negative_index: int
    neutral_index: int
    construct: str
    license_id: str
    required_files: tuple[str, ...]


MODEL_CONTRACTS = {
    DISTIL_BACKEND: ModelContract(
        backend_id=DISTIL_BACKEND,
        repo_id="achen0525/DistilBERT_FOMC_Classifier",
        revision="9c061b4ce901418ff7839e9d89ca07156125d201",
        directory_name=(
            "achen0525--DistilBERT_FOMC_Classifier--"
            "9c061b4ce901418ff7839e9d89ca07156125d201"
        ),
        weights_file="model.safetensors",
        weights_sha256="02a94f213a5fd9f6401ee403afdaf382374a5b20753c84c62c1224ea4a79075a",
        readme_sha256="50f9d60fc0449a5e7010fecbe0ac8758628cc6a27b8c6019153f8266a5d7d7f5",
        config_sha256="145dce956b4f0512e98ab6459810fe5adc2cbf9dd19fb57d46078574b942ac3e",
        labels=("dovish", "hawkish", "neutral"),
        positive_index=1,
        negative_index=0,
        neutral_index=2,
        construct="monetary_policy_stance_hawkish_minus_dovish",
        license_id="apache-2.0",
        required_files=(
            "README.md",
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.txt",
        ),
    ),
    FINBERT_BACKEND: ModelContract(
        backend_id=FINBERT_BACKEND,
        repo_id="ProsusAI/finbert",
        revision="4556d13015211d73dccd3fdd39d39232506f3e43",
        directory_name="ProsusAI--finbert--4556d13015211d73dccd3fdd39d39232506f3e43",
        weights_file="pytorch_model.bin",
        weights_sha256="e15a7b5738df7f17553399b6d94c6e2ff69c89245d066e8e5d183f5803a554e3",
        readme_sha256="23bf85400efe1e88cbe2bf79cec8c5ff81fe9806c599f3ca85e18288ab871008",
        config_sha256="f6449ddda85eb726207a40be59c0cd3bd4b142ccb27298d5e45f9ae3396b1abe",
        labels=("positive", "negative", "neutral"),
        positive_index=0,
        negative_index=1,
        neutral_index=2,
        construct="financial_valence_positive_minus_negative",
        license_id="not_declared_in_model_card",
        required_files=(
            "README.md",
            "config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.txt",
        ),
    ),
}

# Phrase-level terms avoid treating every generic increase/decrease as a policy
# signal.  This is a transparent operational proxy, not a literal Factiva PMI
# replication of Lucca and Trebbi (2009).
LEXICON_PATTERNS = {
    "hawkish": (
        ("hawkish", r"\bhawkish\b"),
        ("tighten", r"\btighten(?:s|ed|ing)?\b"),
        ("restrictive", r"\brestrictive\b"),
        ("raise_rates", r"\brais(?:e|es|ed|ing) (?:the )?(?:policy )?rates?\b"),
        ("rate_hike", r"\brate hikes?\b"),
        ("hike_rates", r"\bhik(?:e|es|ed|ing) (?:the )?rates?\b"),
        ("higher_policy_rate", r"\bhigher (?:policy |interest )rates?\b"),
        ("additional_firming", r"\badditional (?:policy )?firming\b"),
        ("remove_accommodation", r"\bremov(?:e|es|ed|ing) (?:policy )?accommodation\b"),
    ),
    "dovish": (
        ("dovish", r"\bdovish\b"),
        ("ease", r"\beas(?:e|es|ed|ing)\b"),
        ("accommodative", r"\baccommodative\b"),
        ("cut_rates", r"\bcut(?:s|ting)? (?:the )?(?:policy )?rates?\b"),
        ("rate_cut", r"\brate cuts?\b"),
        ("lower_policy_rate", r"\blower (?:policy |interest )rates?\b"),
        ("policy_accommodation", r"\bpolicy accommodation\b"),
        (
            "provide_accommodation",
            r"\bprovid(?:e|es|ed|ing) (?:additional )?accommodation\b",
        ),
    ),
}
WORD_RE = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b")
COMPILED_LEXICON = {
    side: tuple((name, re.compile(pattern, re.IGNORECASE)) for name, pattern in terms)
    for side, terms in LEXICON_PATTERNS.items()
}


class SentimentScoringError(RuntimeError):
    """The sentiment scoring contract or a sealed artifact is invalid."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _record_sha256(value: Mapping[str, Any]) -> str:
    return sha256_text(_canonical(dict(value)))


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    return sha256_text(_canonical(list(token_ids)))


def _binding(
    path: Path,
    *,
    rows: int | None = None,
    payload_sha256: str | None = None,
) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise SentimentScoringError(f"bound path is a symlink: {unresolved}")
    path = unresolved.resolve()
    if not path.is_file():
        raise SentimentScoringError(f"bound file is missing: {path}")
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    if payload_sha256 is not None:
        result["payload_sha256"] = payload_sha256
    return result


def _implementation_binding() -> dict[str, Any]:
    binding = _binding(IMPLEMENTATION_PATH)
    if binding["sha256"] != IMPLEMENTATION_SHA256_AT_IMPORT:
        raise SentimentScoringError(
            "implementation source changed after this process imported it"
        )
    return binding


def _validate_implementation_binding(value: Any) -> None:
    if not isinstance(value, Mapping) or dict(value) != _implementation_binding():
        raise SentimentScoringError("implementation source binding drift")


def _fresh_directory(path: Path) -> Path:
    unresolved = path.expanduser()
    if unresolved.is_symlink() or os.path.lexists(unresolved):
        raise SentimentScoringError(f"output directory must be fresh: {unresolved}")
    path = unresolved.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_readonly_json(path: Path, value: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise SentimentScoringError(f"JSON output is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(_canonical(dict(value)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        if os.path.lexists(path):
            raise SentimentScoringError(f"JSONL output is create-only: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.handle = os.fdopen(fd, "w", encoding="utf-8")
        self.rows = 0

    def write(self, value: Mapping[str, Any]) -> None:
        self.handle.write(_canonical(dict(value)) + "\n")
        self.rows += 1
        if self.rows % FSYNC_EVERY_ROWS == 0:
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def close(self) -> None:
        if self.handle.closed:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.path.chmod(0o444)
        _fsync_directory(self.path.parent)

    def __enter__(self) -> _JsonlWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _read_json(path: Path, *, sealed: bool = True) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise SentimentScoringError(f"JSON is missing or a symlink: {unresolved}")
    try:
        value = json.loads(unresolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SentimentScoringError(f"invalid JSON {unresolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise SentimentScoringError(f"JSON root is not an object: {unresolved}")
    if sealed:
        try:
            validate_manifest_integrity(value)
        except Exception as exc:
            raise SentimentScoringError(
                f"sealed JSON integrity failed for {unresolved}: {exc}"
            ) from exc
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    unresolved = path.expanduser()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise SentimentScoringError(f"JSONL is missing or a symlink: {unresolved}")
    payload = unresolved.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise SentimentScoringError(f"JSONL has an incomplete tail: {unresolved}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(keepends=True), 1):
        try:
            text = raw[:-1].decode("utf-8")
            value = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SentimentScoringError(
                f"invalid JSONL row {unresolved}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict) or (_canonical(value) + "\n").encode() != raw:
            raise SentimentScoringError(
                f"noncanonical JSONL row: {unresolved}:{line_number}"
            )
        rows.append(value)
    return rows


def _role(row: Mapping[str, Any]) -> str:
    for key in ("original_post_split_role", "source_split", "original_qa_split"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise SentimentScoringError("source row has no regression role")


def _inventory_row(
    row: Mapping[str, Any],
    *,
    arm: str,
    replicate_id: int | None,
    text: str,
    text_sha256: str,
    source_generation_line_number: int,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    text_id = (
        f"reference::{sample_id}"
        if replicate_id is None
        else f"{arm}::{sample_id}::replicate-{replicate_id:02d}"
    )
    return {
        "schema_version": INVENTORY_SCHEMA,
        "text_id": text_id,
        "arm": arm,
        "sample_id": sample_id,
        "meeting_id": str(row["meeting_id"]),
        "meeting_date": str(row["meeting_end_date"]),
        "meeting_start_date": str(row["meeting_start_date"]),
        "era": str(row["era"]),
        "role": _role(row),
        "topic": str(row["topic"]),
        "topic_order": int(row["topic_order"]),
        "replicate_id": replicate_id,
        "replicate_seed": (
            None if replicate_id is None else int(row["replicate_seed"])
        ),
        "cp318_selection_exposed": bool(row["cp318_selection_exposed"]),
        "text": text,
        "text_sha256": text_sha256,
        "empty": not bool(text.strip()),
        "source_generation_line_number": source_generation_line_number,
        "source_generation_model_id": str(row["model_id"]),
        "source_generation_completion_sha256": str(row["completion_sha256"]),
        "source_reference_minutes_sha256": str(row["reference_minutes_sha256"]),
        "source_answer_sha256": str(row["answer_sha256"]),
    }


def _assemble_inventory_rows(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    expected_keys: list[tuple[str, int]] | None = None
    normalized: dict[str, list[Mapping[str, Any]]] = {}
    for model_id in GENERATION_MODEL_ORDER:
        rows = rows_by_model.get(model_id)
        if rows is None or len(rows) != EXPECTED_GENERATED_TOPICS_PER_ARM:
            raise SentimentScoringError(
                f"{model_id} source coverage is not {EXPECTED_GENERATED_TOPICS_PER_ARM:,}"
            )
        keys = [
            (str(row.get("sample_id")), int(row.get("replicate_id", -1)))
            for row in rows
        ]
        if len(set(keys)) != len(keys):
            raise SentimentScoringError(f"{model_id} tuple keys are not unique")
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise SentimentScoringError(
                "generation arms do not share exact tuple order"
            )
        normalized[model_id] = list(rows)

    assert expected_keys is not None
    base_rows = normalized["chk0"]
    reference_rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for line_number, row in enumerate(base_rows, 1):
        sample_id = str(row["sample_id"])
        if sample_id in seen_samples:
            continue
        seen_samples.add(sample_id)
        if int(row["replicate_id"]) != 0:
            raise SentimentScoringError("first sample occurrence is not replicate zero")
        reference = str(row["reference_minutes"])
        if sha256_text(reference) != row["reference_minutes_sha256"]:
            raise SentimentScoringError("reference text/hash drift")
        reference_rows.append(
            _inventory_row(
                row,
                arm="reference",
                replicate_id=None,
                text=reference,
                text_sha256=str(row["reference_minutes_sha256"]),
                source_generation_line_number=line_number,
            )
        )
    if len(reference_rows) != EXPECTED_REFERENCE_TOPICS:
        raise SentimentScoringError("reference topic coverage is not N=2,048")

    result = list(reference_rows)
    for model_id in GENERATION_MODEL_ORDER:
        for line_number, row in enumerate(normalized[model_id], 1):
            replicate_id = int(row["replicate_id"])
            answer = str(row["answer"])
            if row.get("model_id") != model_id or sha256_text(answer) != row.get(
                "answer_sha256"
            ):
                raise SentimentScoringError(f"{model_id} answer identity drift")
            counterpart = base_rows[line_number - 1]
            for key in (
                "sample_id",
                "meeting_id",
                "meeting_end_date",
                "meeting_start_date",
                "era",
                "topic",
                "topic_order",
                "replicate_id",
                "replicate_seed",
                "cp318_selection_exposed",
                "reference_minutes_sha256",
            ):
                if row.get(key) != counterpart.get(key):
                    raise SentimentScoringError(
                        f"cross-arm source metadata drift at {model_id}:{line_number}:{key}"
                    )
            result.append(
                _inventory_row(
                    row,
                    arm=model_id,
                    replicate_id=replicate_id,
                    text=answer,
                    text_sha256=str(row["answer_sha256"]),
                    source_generation_line_number=line_number,
                )
            )
    _validate_inventory_rows(result)
    return result


def _validate_inventory_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != EXPECTED_TEXTS:
        raise SentimentScoringError(f"text inventory is not N={EXPECTED_TEXTS:,}")
    if len({str(row.get("text_id")) for row in rows}) != len(rows):
        raise SentimentScoringError("text IDs are not unique")
    arm_counts = Counter(str(row.get("arm")) for row in rows)
    expected_counts = {
        "reference": EXPECTED_REFERENCE_TOPICS,
        **{arm: EXPECTED_GENERATED_TOPICS_PER_ARM for arm in GENERATION_MODEL_ORDER},
    }
    if arm_counts != expected_counts:
        raise SentimentScoringError(f"text arm coverage drift: {arm_counts}")
    group_topics: dict[tuple[str, str, int | None], list[int]] = defaultdict(list)
    group_exposures: dict[tuple[str, str, int | None], set[bool]] = defaultdict(set)
    for row in rows:
        text = row.get("text")
        if (
            row.get("schema_version") != INVENTORY_SCHEMA
            or row.get("arm") not in ARM_ORDER
            or not isinstance(text, str)
            or row.get("text_sha256") != sha256_text(text)
            or row.get("empty") is not (not bool(text.strip()))
            or row.get("topic") not in data_contract.CORE_TOPICS
            or row.get("topic_order")
            != data_contract.CORE_TOPICS.index(str(row.get("topic")))
            or not isinstance(row.get("cp318_selection_exposed"), bool)
        ):
            raise SentimentScoringError("text inventory row contract drift")
        replicate_id = row.get("replicate_id")
        if row.get("arm") == "reference":
            if replicate_id is not None or row.get("replicate_seed") is not None:
                raise SentimentScoringError("reference unexpectedly has a replicate")
        elif (
            isinstance(replicate_id, bool)
            or not isinstance(replicate_id, int)
            or replicate_id not in range(REPLICATE_COUNT)
            or row.get("replicate_seed") != preparation.REPLICATE_SEEDS[replicate_id]
        ):
            raise SentimentScoringError("synthetic replicate identity drift")
        group_key = (str(row["arm"]), str(row["meeting_id"]), replicate_id)
        group_topics[group_key].append(int(row["topic_order"]))
        group_exposures[group_key].add(bool(row["cp318_selection_exposed"]))
    if len(group_topics) != EXPECTED_MEETING_ROWS or any(
        sorted(values) != list(range(TOPIC_COUNT)) for values in group_topics.values()
    ):
        raise SentimentScoringError("inventory meeting/Core8 closure drift")
    if any(len(values) != 1 for values in group_exposures.values()):
        raise SentimentScoringError(
            "cp318 selection exposure is not constant across a meeting's Core8 topics"
        )


def prepare_inventory(
    *,
    suite_manifest: Path,
    cohort_path: Path,
    cohort_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = _fresh_directory(output_dir)
    try:
        loaded = generation_suite.load_and_validate_suite(
            suite_manifest,
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            expected_scope="formal_merged_panel",
            max_num_seqs=generation_suite.FIXED_MAX_NUM_SEQS,
        )
    except Exception as exc:
        raise SentimentScoringError(
            f"generation suite validation failed: {exc}"
        ) from exc
    rows_by_model = {
        model_id: loaded["runs"][model_id]["results"]
        for model_id in GENERATION_MODEL_ORDER
    }
    rows = _assemble_inventory_rows(rows_by_model)
    inventory_path = output_dir / "text_inventory.v1.jsonl"
    with _JsonlWriter(inventory_path) as writer:
        for row in rows:
            writer.write(row)
    empty_counts = Counter(str(row["arm"]) for row in rows if bool(row.get("empty")))
    manifest = seal_manifest(
        {
            "schema_version": INVENTORY_MANIFEST_SCHEMA,
            "status": "complete",
            "immutable": True,
            "evaluation_id": EVALUATION_ID,
            "source_generation_suite": copy.deepcopy(loaded["manifest_binding"]),
            "source_generation_runs": {
                model_id: copy.deepcopy(loaded["runs"][model_id]["manifest_binding"])
                for model_id in GENERATION_MODEL_ORDER
            },
            "source_cohort": _binding(cohort_path),
            "text_inventory": _binding(inventory_path, rows=EXPECTED_TEXTS),
            "coverage": {
                "arms": list(ARM_ORDER),
                "meetings": data_contract.EXPECTED_MEETINGS,
                "topics_per_meeting": TOPIC_COUNT,
                "replicates": REPLICATE_COUNT,
                "texts": EXPECTED_TEXTS,
                "arm_texts": {
                    "reference": EXPECTED_REFERENCE_TOPICS,
                    **{
                        arm: EXPECTED_GENERATED_TOPICS_PER_ARM
                        for arm in GENERATION_MODEL_ORDER
                    },
                },
                "empty_texts": {
                    arm: int(empty_counts.get(arm, 0)) for arm in ARM_ORDER
                },
            },
            "reference_semantics": {
                "arm_label": "reference",
                "reference_type": "deterministic_source_grounded_minutes_style_v1",
                "is_genuine_official_minutes": False,
                "official_minutes_role": "secondary_context_not_model_input",
            },
            "implementation": _implementation_binding(),
        }
    )
    _write_readonly_json(output_dir / "manifest.json", manifest)
    return manifest


def load_and_validate_inventory(
    manifest_path: Path, *, deep_sources: bool = False
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    payload_sha = validate_manifest_integrity(manifest)
    if (
        manifest.get("schema_version") != INVENTORY_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("evaluation_id") != EVALUATION_ID
    ):
        raise SentimentScoringError("inventory manifest header drift")
    _validate_implementation_binding(manifest.get("implementation"))
    inventory_binding = manifest.get("text_inventory")
    if not isinstance(inventory_binding, Mapping):
        raise SentimentScoringError("inventory file binding is missing")
    path = Path(str(inventory_binding.get("path")))
    if dict(inventory_binding) != _binding(path, rows=EXPECTED_TEXTS):
        raise SentimentScoringError("inventory file binding drift")
    rows = _read_jsonl(path)
    _validate_inventory_rows(rows)
    if deep_sources:
        source_suite = manifest.get("source_generation_suite")
        cohort = manifest.get("source_cohort")
        if not isinstance(source_suite, Mapping) or not isinstance(cohort, Mapping):
            raise SentimentScoringError("inventory source bindings are missing")
        cohort_path = Path(str(cohort.get("path")))
        if dict(cohort) != _binding(cohort_path):
            raise SentimentScoringError("inventory cohort binding drift")
        try:
            loaded = generation_suite.load_and_validate_suite(
                Path(str(source_suite.get("path"))),
                cohort_path=cohort_path,
                cohort_sha256=str(cohort["sha256"]),
                expected_scope="formal_merged_panel",
                max_num_seqs=generation_suite.FIXED_MAX_NUM_SEQS,
            )
        except Exception as exc:
            raise SentimentScoringError(
                f"deep source suite validation failed: {exc}"
            ) from exc
        if source_suite != loaded["manifest_binding"]:
            raise SentimentScoringError("generation suite binding drift")
        rebuilt = _assemble_inventory_rows(
            {
                model_id: loaded["runs"][model_id]["results"]
                for model_id in GENERATION_MODEL_ORDER
            }
        )
        if rows != rebuilt:
            raise SentimentScoringError("inventory is not an exact source reshape")
    return {
        "manifest": manifest,
        "manifest_binding": _binding(manifest_path, payload_sha256=payload_sha),
        "rows": rows,
    }


def _model_dir(contract: ModelContract, model_root: Path) -> Path:
    return model_root / contract.directory_name


def download_and_seal_model(*, backend_id: str, model_root: Path) -> dict[str, Any]:
    contract = MODEL_CONTRACTS.get(backend_id)
    if contract is None:
        raise SentimentScoringError(f"backend has no downloadable model: {backend_id}")
    model_root = model_root.expanduser().resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    model_dir = model_root / contract.directory_name
    if model_dir.is_symlink():
        raise SentimentScoringError("model output directory is a symlink")
    manifest_path = model_dir / "model_manifest.json"
    if manifest_path.exists():
        return load_and_validate_model_manifest(manifest_path)["manifest"]
    if model_dir.exists() and any(model_dir.iterdir()):
        raise SentimentScoringError("unsealed model directory is not fresh")
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=contract.repo_id,
            revision=contract.revision,
            local_dir=model_dir,
            allow_patterns=list(contract.required_files),
        )
    except Exception as exc:
        raise SentimentScoringError(f"model download failed: {exc}") from exc
    resolved_path = model_dir / "RESOLVED_REVISION"
    resolved_path.write_text(contract.revision + "\n", encoding="utf-8")
    with resolved_path.open("r+", encoding="utf-8") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    files: dict[str, Any] = {}
    for relative in contract.required_files:
        path = model_dir / relative
        if path.is_symlink() or not path.is_file():
            raise SentimentScoringError(f"downloaded model file missing: {relative}")
        files[relative] = _binding(path)
        path.chmod(0o444)
    resolved_path.chmod(0o444)
    manifest = seal_manifest(
        {
            "schema_version": MODEL_MANIFEST_SCHEMA,
            "status": "complete",
            "immutable": True,
            "backend_id": backend_id,
            "repo_id": contract.repo_id,
            "revision": contract.revision,
            "license": contract.license_id,
            "construct": contract.construct,
            "label_order": list(contract.labels),
            "signed_score": {
                "positive_index": contract.positive_index,
                "negative_index": contract.negative_index,
                "neutral_index": contract.neutral_index,
                "formula": "p[positive_index]-p[negative_index]",
            },
            "resolved_revision": _binding(resolved_path),
            "files": files,
            "implementation": _implementation_binding(),
        }
    )
    _write_readonly_json(manifest_path, manifest)
    return manifest


def load_and_validate_model_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    payload_sha = validate_manifest_integrity(manifest)
    backend_id = str(manifest.get("backend_id") or "")
    contract = MODEL_CONTRACTS.get(backend_id)
    if contract is None:
        raise SentimentScoringError("model manifest backend is unsupported")
    model_dir = manifest_path.resolve().parent
    resolved_path = model_dir / "RESOLVED_REVISION"
    if (
        manifest.get("schema_version") != MODEL_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("repo_id") != contract.repo_id
        or manifest.get("revision") != contract.revision
        or manifest.get("construct") != contract.construct
        or manifest.get("label_order") != list(contract.labels)
        or not resolved_path.is_file()
        or resolved_path.read_text(encoding="utf-8") != contract.revision + "\n"
        or manifest.get("resolved_revision") != _binding(resolved_path)
    ):
        raise SentimentScoringError("model manifest contract drift")
    _validate_implementation_binding(manifest.get("implementation"))
    expected_files = {
        relative: _binding(model_dir / relative) for relative in contract.required_files
    }
    if manifest.get("files") != expected_files:
        raise SentimentScoringError("model file bindings drift")
    if (
        expected_files[contract.weights_file]["sha256"] != contract.weights_sha256
        or expected_files["README.md"]["sha256"] != contract.readme_sha256
        or expected_files["config.json"]["sha256"] != contract.config_sha256
    ):
        raise SentimentScoringError("model frozen upstream hashes drift")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if backend_id == FINBERT_BACKEND:
        expected_id2label = {"0": "positive", "1": "negative", "2": "neutral"}
        if config.get("id2label") != expected_id2label:
            raise SentimentScoringError("FinBERT config label mapping drift")
    elif config.get("id2label") != {
        "0": "LABEL_0",
        "1": "LABEL_1",
        "2": "LABEL_2",
    }:
        raise SentimentScoringError("DistilBERT generic config labels drift")
    return {
        "manifest": manifest,
        "manifest_binding": _binding(manifest_path, payload_sha256=payload_sha),
        "model_dir": model_dir,
        "contract": contract,
    }


def _lexicon_score(text: str) -> dict[str, Any]:
    token_count = len(WORD_RE.findall(text))
    matches: dict[str, list[dict[str, Any]]] = {"hawkish": [], "dovish": []}
    for side, patterns in COMPILED_LEXICON.items():
        for term_id, pattern in patterns:
            for match in pattern.finditer(text):
                matches[side].append(
                    {
                        "term_id": term_id,
                        "text": match.group(0),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
        matches[side].sort(
            key=lambda value: (value["start"], value["end"], value["term_id"])
        )
    hawkish = len(matches["hawkish"])
    dovish = len(matches["dovish"])
    score = None if token_count == 0 else (hawkish - dovish) / token_count
    return {
        "token_count": token_count,
        "hawkish_count": hawkish,
        "dovish_count": dovish,
        "matches": matches,
        "score": score,
    }


def _topic_base(inventory: Mapping[str, Any], *, backend_id: str) -> dict[str, Any]:
    return {
        "schema_version": TOPIC_SCHEMA,
        "backend": backend_id,
        "construct": (
            "monetary_policy_stance_hawkish_minus_dovish"
            if backend_id in {LEXICON_BACKEND, DISTIL_BACKEND}
            else "financial_valence_positive_minus_negative"
        ),
        **{
            key: copy.deepcopy(inventory[key])
            for key in (
                "text_id",
                "arm",
                "sample_id",
                "meeting_id",
                "meeting_date",
                "meeting_start_date",
                "era",
                "role",
                "topic",
                "topic_order",
                "replicate_id",
                "replicate_seed",
                "cp318_selection_exposed",
                "text_sha256",
                "empty",
            )
        },
    }


def _aggregate_meeting_rows(
    topic_rows: Sequence[Mapping[str, Any]], *, backend_id: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int | None], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in topic_rows:
        grouped[
            (str(row["arm"]), str(row["meeting_id"]), row.get("replicate_id"))
        ].append(row)
    rows: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        arm_keys = sorted(
            (key for key in grouped if key[0] == arm),
            key=lambda key: (key[1], -1 if key[2] is None else int(key[2])),
        )
        for key in arm_keys:
            topics = sorted(grouped[key], key=lambda value: int(value["topic_order"]))
            if len(topics) != TOPIC_COUNT or [row["topic"] for row in topics] != list(
                data_contract.CORE_TOPICS
            ):
                raise SentimentScoringError(f"meeting is not exact Core8: {key}")
            first = topics[0]
            for field in (
                "meeting_date",
                "meeting_start_date",
                "era",
                "role",
                "cp318_selection_exposed",
                "replicate_id",
                "replicate_seed",
            ):
                if any(row.get(field) != first.get(field) for row in topics[1:]):
                    raise SentimentScoringError(
                        f"meeting metadata drift: {key}:{field}"
                    )
            scores = [row.get("score") for row in topics]
            complete = all(isinstance(score, (float, int)) for score in scores)
            neutral_scores = [
                float(score) if isinstance(score, (float, int)) else 0.0
                for score in scores
            ]
            topic_hashes = [str(row["text_sha256"]) for row in topics]
            topic_row_hashes = [_record_sha256(row) for row in topics]
            rows.append(
                {
                    "schema_version": MEETING_SCHEMA,
                    "meeting_id": str(first["meeting_id"]),
                    "meeting_date": str(first["meeting_date"]),
                    "meeting_start_date": str(first["meeting_start_date"]),
                    "era": str(first["era"]),
                    "role": str(first["role"]),
                    "cp318_selection_exposed": bool(first["cp318_selection_exposed"]),
                    "backend": backend_id,
                    "construct": str(first["construct"]),
                    "arm": arm,
                    "replicate_id": first.get("replicate_id"),
                    "replicate_seed": first.get("replicate_seed"),
                    "score": (
                        sum(float(score) for score in scores) / TOPIC_COUNT
                        if complete
                        else None
                    ),
                    "complete_core8": complete,
                    "neutral_imputed_score": sum(neutral_scores) / TOPIC_COUNT,
                    "missing_topics": [
                        str(row["topic"])
                        for row, score in zip(topics, scores, strict=True)
                        if score is None
                    ],
                    "source_hashes": {
                        "topic_text_sha256s": topic_hashes,
                        "topic_score_row_sha256s": topic_row_hashes,
                    },
                }
            )
    if len(rows) != EXPECTED_MEETING_ROWS:
        raise SentimentScoringError(
            f"meeting score coverage is not N={EXPECTED_MEETING_ROWS:,}"
        )
    return rows


def _backend_manifest(
    *,
    backend_id: str,
    output_dir: Path,
    inventory_binding: Mapping[str, Any],
    window_path: Path,
    window_rows: int,
    topic_path: Path,
    topic_rows: Sequence[Mapping[str, Any]],
    meeting_path: Path,
    meeting_rows: Sequence[Mapping[str, Any]],
    model_binding: Mapping[str, Any] | None,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    complete_counts = Counter(
        str(row["arm"]) for row in meeting_rows if bool(row["complete_core8"])
    )
    return seal_manifest(
        {
            "schema_version": BACKEND_MANIFEST_SCHEMA,
            "status": "complete",
            "immutable": True,
            "evaluation_id": EVALUATION_ID,
            "backend": backend_id,
            "construct": str(meeting_rows[0]["construct"]),
            "inventory_manifest": copy.deepcopy(dict(inventory_binding)),
            "model_manifest": (
                None if model_binding is None else copy.deepcopy(dict(model_binding))
            ),
            "artifacts": {
                "window_scores": _binding(window_path, rows=window_rows),
                "topic_scores": _binding(topic_path, rows=len(topic_rows)),
                "meeting_scores": _binding(meeting_path, rows=len(meeting_rows)),
            },
            "coverage": {
                "texts": len(topic_rows),
                "window_rows": window_rows,
                "topic_rows": len(topic_rows),
                "meeting_rows": len(meeting_rows),
                "complete_core8_by_arm": {
                    arm: int(complete_counts.get(arm, 0)) for arm in ARM_ORDER
                },
                "incomplete_core8_by_arm": {
                    arm: (
                        data_contract.EXPECTED_MEETINGS
                        * (1 if arm == "reference" else REPLICATE_COUNT)
                        - int(complete_counts.get(arm, 0))
                    )
                    for arm in ARM_ORDER
                },
            },
            "aggregation": {
                "topic_weighting": "equal_1_over_8",
                "replicate_weighting": "equal_1_over_5_after_topic_aggregation",
                "empty_primary": "missing_incomplete_core8",
                "empty_sensitivity": "signed_zero_equivalent_to_neutral_probability_one",
                "neural_content_tokens": CONTENT_TOKENS,
                "neural_overlap_tokens": OVERLAP_TOKENS,
                "neural_window_step": WINDOW_STEP,
                "neural_overlap_correction": "sum_inverse_token_coverage",
            },
            "runtime": copy.deepcopy(dict(runtime)),
            "implementation": _implementation_binding(),
        }
    )


def score_lexicon(*, inventory_manifest: Path, output_dir: Path) -> dict[str, Any]:
    inventory = load_and_validate_inventory(inventory_manifest)
    output_dir = _fresh_directory(output_dir)
    window_path = output_dir / "window_scores.jsonl"
    topic_path = output_dir / "topic_scores.jsonl"
    meeting_path = output_dir / "meeting_scores.jsonl"
    topic_rows: list[dict[str, Any]] = []
    with (
        _JsonlWriter(window_path) as window_writer,
        _JsonlWriter(topic_path) as topic_writer,
    ):
        for source in inventory["rows"]:
            result = _lexicon_score(str(source["text"]))
            if result["token_count"]:
                window_writer.write(
                    {
                        "schema_version": WINDOW_SCHEMA,
                        "backend": LEXICON_BACKEND,
                        "text_id": source["text_id"],
                        "text_sha256": source["text_sha256"],
                        "window_index": 0,
                        "character_start": 0,
                        "character_end": len(str(source["text"])),
                        "token_count": result["token_count"],
                        "hawkish_count": result["hawkish_count"],
                        "dovish_count": result["dovish_count"],
                        "matches": result["matches"],
                        "score": result["score"],
                        "aggregation_weight": 1.0,
                    }
                )
            topic = {
                **_topic_base(source, backend_id=LEXICON_BACKEND),
                "window_count": 0 if result["token_count"] == 0 else 1,
                "token_count": result["token_count"],
                "hawkish_count": result["hawkish_count"],
                "dovish_count": result["dovish_count"],
                "score": result["score"],
                "neutral_imputed_score": (
                    0.0 if result["score"] is None else result["score"]
                ),
            }
            topic_writer.write(topic)
            topic_rows.append(topic)
    meeting_rows = _aggregate_meeting_rows(topic_rows, backend_id=LEXICON_BACKEND)
    with _JsonlWriter(meeting_path) as writer:
        for row in meeting_rows:
            writer.write(row)
    manifest = _backend_manifest(
        backend_id=LEXICON_BACKEND,
        output_dir=output_dir,
        inventory_binding=inventory["manifest_binding"],
        window_path=window_path,
        window_rows=sum(1 for row in topic_rows if row["window_count"] == 1),
        topic_path=topic_path,
        topic_rows=topic_rows,
        meeting_path=meeting_path,
        meeting_rows=meeting_rows,
        model_binding=None,
        runtime={
            "device": "cpu",
            "lexicon_formula": "(hawkish_phrase_count-dovish_phrase_count)/word_count",
            "lexicon_patterns": {
                side: [
                    {"term_id": term_id, "regex": pattern}
                    for term_id, pattern in patterns
                ]
                for side, patterns in LEXICON_PATTERNS.items()
            },
            "literal_lucca_trebbi_replication": False,
        },
    )
    _write_readonly_json(output_dir / "manifest.json", manifest)
    return manifest


@dataclass(frozen=True)
class WindowSpec:
    index: int
    token_start: int
    token_end: int
    content_ids: tuple[int, ...]
    input_ids: tuple[int, ...]
    aggregation_weight: float


def _window_specs(tokenizer: Any, text: str) -> list[WindowSpec]:
    content_ids = list(tokenizer.encode(text, add_special_tokens=False))
    if not content_ids:
        return []
    if int(tokenizer.model_max_length) != 512:
        raise SentimentScoringError("neural tokenizer model_max_length is not 512")
    if int(tokenizer.num_special_tokens_to_add(pair=False)) != 2:
        raise SentimentScoringError("neural tokenizer special-token count is not two")
    starts = [0]
    while starts[-1] + CONTENT_TOKENS < len(content_ids):
        starts.append(starts[-1] + WINDOW_STEP)
    spans = [(start, min(start + CONTENT_TOKENS, len(content_ids))) for start in starts]
    coverage = [0] * len(content_ids)
    for start, end in spans:
        for index in range(start, end):
            coverage[index] += 1
    if any(value <= 0 for value in coverage):
        raise SentimentScoringError("windowing did not cover every content token")
    specs: list[WindowSpec] = []
    for index, (start, end) in enumerate(spans):
        ids = content_ids[start:end]
        input_ids = list(tokenizer.build_inputs_with_special_tokens(ids))
        if len(input_ids) > 512:
            raise SentimentScoringError("neural window exceeds 512 input tokens")
        weight = sum(1.0 / coverage[token_index] for token_index in range(start, end))
        specs.append(
            WindowSpec(
                index=index,
                token_start=start,
                token_end=end,
                content_ids=tuple(ids),
                input_ids=tuple(input_ids),
                aggregation_weight=weight,
            )
        )
    if not math.isclose(
        sum(spec.aggregation_weight for spec in specs),
        len(content_ids),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise SentimentScoringError("coverage-corrected weights do not sum to tokens")
    return specs


def score_neural(
    *,
    backend_id: str,
    inventory_manifest: Path,
    model_manifest: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    expected_visible_gpu: str | None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise SentimentScoringError("batch size must be positive")
    inventory = load_and_validate_inventory(inventory_manifest)
    model_loaded = load_and_validate_model_manifest(model_manifest)
    contract: ModelContract = model_loaded["contract"]
    if backend_id != contract.backend_id:
        raise SentimentScoringError("backend/model manifest mismatch")
    output_dir = _fresh_directory(output_dir)
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SentimentScoringError(f"neural dependencies unavailable: {exc}") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SentimentScoringError("CUDA device requested but CUDA is unavailable")
    visible_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device.startswith("cuda") and (
        expected_visible_gpu is None or visible_cuda_devices != expected_visible_gpu
    ):
        raise SentimentScoringError(
            "CUDA scoring requires an exact --expected-visible-gpu binding"
        )
    torch.manual_seed(0)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_loaded["model_dir"], local_files_only=True, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_loaded["model_dir"], local_files_only=True
    )
    model.eval()
    model.to(device)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    pending: list[tuple[Mapping[str, Any], WindowSpec]] = []
    specs_by_text: dict[str, list[WindowSpec]] = {}
    for source in inventory["rows"]:
        specs = _window_specs(tokenizer, str(source["text"]))
        specs_by_text[str(source["text_id"])] = specs
        pending.extend((source, spec) for spec in specs)

    window_path = output_dir / "window_scores.jsonl"
    topic_path = output_dir / "topic_scores.jsonl"
    meeting_path = output_dir / "meeting_scores.jsonl"
    accumulators: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"weight": 0.0, "probability_sums": [0.0, 0.0, 0.0]}
    )
    with _JsonlWriter(window_path) as writer, torch.inference_mode():
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            padded = tokenizer.pad(
                {"input_ids": [list(spec.input_ids) for _, spec in batch]},
                padding=True,
                return_tensors="pt",
            )
            padded = {key: value.to(device) for key, value in padded.items()}
            logits_tensor = model(**padded).logits.float()
            probabilities_tensor = torch.softmax(logits_tensor, dim=-1)
            logits_batch = logits_tensor.cpu().tolist()
            probabilities_batch = probabilities_tensor.cpu().tolist()
            for (source, spec), logits, probabilities in zip(
                batch, logits_batch, probabilities_batch, strict=True
            ):
                if len(logits) != 3 or len(probabilities) != 3:
                    raise SentimentScoringError(
                        "neural backend did not return 3 labels"
                    )
                score = float(probabilities[contract.positive_index]) - float(
                    probabilities[contract.negative_index]
                )
                text_id = str(source["text_id"])
                accumulator = accumulators[text_id]
                accumulator["weight"] += spec.aggregation_weight
                for label_index, probability in enumerate(probabilities):
                    accumulator["probability_sums"][label_index] += (
                        spec.aggregation_weight * float(probability)
                    )
                writer.write(
                    {
                        "schema_version": WINDOW_SCHEMA,
                        "backend": backend_id,
                        "construct": contract.construct,
                        "text_id": text_id,
                        "text_sha256": source["text_sha256"],
                        "window_index": spec.index,
                        "token_start": spec.token_start,
                        "token_end": spec.token_end,
                        "content_token_count": len(spec.content_ids),
                        "content_token_ids": list(spec.content_ids),
                        "content_token_ids_sha256": _token_ids_sha256(spec.content_ids),
                        "input_token_ids": list(spec.input_ids),
                        "input_token_ids_sha256": _token_ids_sha256(spec.input_ids),
                        "aggregation_weight": spec.aggregation_weight,
                        "logits": [float(value) for value in logits],
                        "probabilities": [float(value) for value in probabilities],
                        "label_order": list(contract.labels),
                        "score": score,
                    }
                )

    topic_rows: list[dict[str, Any]] = []
    with _JsonlWriter(topic_path) as writer:
        for source in inventory["rows"]:
            text_id = str(source["text_id"])
            specs = specs_by_text[text_id]
            accumulator = accumulators.get(text_id)
            if not specs:
                probabilities = None
                score = None
                token_count = 0
                total_weight = 0.0
            else:
                assert accumulator is not None
                token_count = specs[-1].token_end
                total_weight = float(accumulator["weight"])
                if not math.isclose(total_weight, token_count, abs_tol=1e-8):
                    raise SentimentScoringError("topic aggregation weight drift")
                probabilities = [
                    float(value) / total_weight
                    for value in accumulator["probability_sums"]
                ]
                if not math.isclose(sum(probabilities), 1.0, abs_tol=2e-6):
                    raise SentimentScoringError("topic probabilities do not sum to one")
                score = (
                    probabilities[contract.positive_index]
                    - probabilities[contract.negative_index]
                )
            topic = {
                **_topic_base(source, backend_id=backend_id),
                "window_count": len(specs),
                "content_token_count": token_count,
                "coverage_corrected_weight": total_weight,
                "probabilities": probabilities,
                "label_order": list(contract.labels),
                "score": score,
                "neutral_imputed_score": 0.0 if score is None else score,
            }
            writer.write(topic)
            topic_rows.append(topic)
    meeting_rows = _aggregate_meeting_rows(topic_rows, backend_id=backend_id)
    with _JsonlWriter(meeting_path) as writer:
        for row in meeting_rows:
            writer.write(row)
    manifest = _backend_manifest(
        backend_id=backend_id,
        output_dir=output_dir,
        inventory_binding=inventory["manifest_binding"],
        window_path=window_path,
        window_rows=len(pending),
        topic_path=topic_path,
        topic_rows=topic_rows,
        meeting_path=meeting_path,
        meeting_rows=meeting_rows,
        model_binding=model_loaded["manifest_binding"],
        runtime={
            "device": device,
            "expected_visible_gpu": expected_visible_gpu,
            "cuda_visible_devices": visible_cuda_devices,
            "cuda_device_name": (
                torch.cuda.get_device_name(device)
                if device.startswith("cuda")
                else None
            ),
            "batch_size": batch_size,
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "model_dtype": str(next(model.parameters()).dtype),
            "inference_mode": True,
            "eval_mode": not model.training,
            "tf32": False,
            "deterministic_algorithms": True,
        },
    )
    _write_readonly_json(output_dir / "manifest.json", manifest)
    return manifest


def _validate_backend_topic_rows(
    rows: Sequence[Mapping[str, Any]], *, backend_id: str
) -> None:
    if len(rows) != EXPECTED_TEXTS:
        raise SentimentScoringError("backend topic coverage drift")
    if len({str(row.get("text_id")) for row in rows}) != len(rows):
        raise SentimentScoringError("backend topic IDs are not unique")
    for row in rows:
        score = row.get("score")
        if (
            row.get("schema_version") != TOPIC_SCHEMA
            or row.get("backend") != backend_id
            or (
                score is not None
                and (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                    or not -1.0 <= float(score) <= 1.0
                )
            )
            or row.get("neutral_imputed_score") != (0.0 if score is None else score)
        ):
            raise SentimentScoringError("backend topic row drift")


def load_and_validate_meeting_scores(manifest_path: Path) -> dict[str, Any]:
    """Deep-validate a backend and expose regression-ready meeting rows."""

    manifest = _read_json(manifest_path)
    payload_sha = validate_manifest_integrity(manifest)
    backend_id = str(manifest.get("backend") or "")
    if (
        manifest.get("schema_version") != BACKEND_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("evaluation_id") != EVALUATION_ID
        or backend_id not in BACKEND_ORDER
    ):
        raise SentimentScoringError("backend manifest header drift")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise SentimentScoringError("backend artifact bindings missing")
    loaded_artifacts: dict[str, list[dict[str, Any]]] = {}
    for key, expected_rows in (
        ("window_scores", int(manifest["coverage"]["window_rows"])),
        ("topic_scores", EXPECTED_TEXTS),
        ("meeting_scores", EXPECTED_MEETING_ROWS),
    ):
        binding = artifacts.get(key)
        if not isinstance(binding, Mapping):
            raise SentimentScoringError(f"backend {key} binding missing")
        path = Path(str(binding.get("path")))
        if dict(binding) != _binding(path, rows=expected_rows):
            raise SentimentScoringError(f"backend {key} binding drift")
        rows = _read_jsonl(path)
        if len(rows) != expected_rows:
            raise SentimentScoringError(f"backend {key} row count drift")
        loaded_artifacts[key] = rows
    topic_rows = loaded_artifacts["topic_scores"]
    _validate_backend_topic_rows(topic_rows, backend_id=backend_id)
    expected_meetings = _aggregate_meeting_rows(topic_rows, backend_id=backend_id)
    if loaded_artifacts["meeting_scores"] != expected_meetings:
        raise SentimentScoringError("meeting scores are not an exact topic aggregation")
    inventory_binding = manifest.get("inventory_manifest")
    if not isinstance(inventory_binding, Mapping):
        raise SentimentScoringError("backend inventory binding missing")
    inventory = load_and_validate_inventory(Path(str(inventory_binding.get("path"))))
    if inventory_binding != inventory["manifest_binding"]:
        raise SentimentScoringError("backend inventory binding drift")
    inventory_by_id = {str(row["text_id"]): row for row in inventory["rows"]}
    source_bound_fields = (
        "arm",
        "sample_id",
        "meeting_id",
        "meeting_date",
        "meeting_start_date",
        "era",
        "role",
        "topic",
        "topic_order",
        "replicate_id",
        "replicate_seed",
        "cp318_selection_exposed",
        "text_sha256",
        "empty",
    )
    if any(
        any(
            row.get(field)
            != inventory_by_id.get(str(row.get("text_id")), {}).get(field)
            for field in source_bound_fields
        )
        for row in topic_rows
    ):
        raise SentimentScoringError("backend topic/inventory source binding drift")
    _validate_implementation_binding(manifest.get("implementation"))
    model_binding = manifest.get("model_manifest")
    if backend_id == LEXICON_BACKEND:
        if model_binding is not None:
            raise SentimentScoringError("lexicon unexpectedly has model weights")
    else:
        if not isinstance(model_binding, Mapping):
            raise SentimentScoringError("neural backend model binding missing")
        model_loaded = load_and_validate_model_manifest(
            Path(str(model_binding.get("path")))
        )
        if model_binding != model_loaded["manifest_binding"]:
            raise SentimentScoringError("backend model binding drift")
    return {
        "manifest": manifest,
        "manifest_binding": _binding(manifest_path, payload_sha256=payload_sha),
        "meeting_scores": loaded_artifacts["meeting_scores"],
        "topic_scores": topic_rows,
        "window_scores": loaded_artifacts["window_scores"],
    }


def _summaries(
    backends: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for backend_id, loaded in backends.items():
        rows = loaded["meeting_scores"]
        by_arm_meeting: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            by_arm_meeting[str(row["arm"])][str(row["meeting_id"])].append(row)
        arm_means: dict[str, float] = {}
        arm_neutral_means: dict[str, float] = {}
        meeting_values: dict[str, dict[str, float]] = defaultdict(dict)
        for arm in ARM_ORDER:
            for meeting_id, cells in by_arm_meeting[arm].items():
                complete_scores = [
                    float(row["score"]) for row in cells if row["score"] is not None
                ]
                if complete_scores:
                    meeting_values[arm][meeting_id] = sum(complete_scores) / len(
                        complete_scores
                    )
            values = list(meeting_values[arm].values())
            if len(values) != data_contract.EXPECTED_MEETINGS:
                raise SentimentScoringError(
                    f"backend {backend_id}:{arm} loses an entire meeting"
                )
            arm_means[arm] = sum(values) / len(values)
            neutral_values = [
                sum(float(row["neutral_imputed_score"]) for row in cells) / len(cells)
                for cells in by_arm_meeting[arm].values()
            ]
            arm_neutral_means[arm] = sum(neutral_values) / len(neutral_values)
        paired: dict[str, float] = {}
        for left, right, label in (
            ("chk3", "chk1", "chk3_minus_chk1"),
            ("chk3", "chk0", "chk3_minus_chk0"),
            ("chk1", "chk0", "chk1_minus_chk0"),
            ("chk0", "reference", "chk0_minus_reference"),
            ("chk1", "reference", "chk1_minus_reference"),
            ("chk3", "reference", "chk3_minus_reference"),
        ):
            meeting_ids = sorted(set(meeting_values[left]) & set(meeting_values[right]))
            paired[label] = sum(
                meeting_values[left][meeting_id] - meeting_values[right][meeting_id]
                for meeting_id in meeting_ids
            ) / len(meeting_ids)
        result[backend_id] = {
            "meeting_equal_arm_means_complete_replicates": arm_means,
            "meeting_equal_arm_means_neutral_imputation": arm_neutral_means,
            "meeting_equal_paired_differences_complete_replicates": paired,
        }
    return result


def seal_sentiment_suite(
    *,
    inventory_manifest: Path,
    backend_manifests: Mapping[str, Path],
    output_path: Path,
    deep_sources: bool,
) -> dict[str, Any]:
    inventory = load_and_validate_inventory(
        inventory_manifest, deep_sources=deep_sources
    )
    backends = {
        backend_id: load_and_validate_meeting_scores(backend_manifests[backend_id])
        for backend_id in BACKEND_ORDER
    }
    if any(
        loaded["manifest"]["inventory_manifest"] != inventory["manifest_binding"]
        for loaded in backends.values()
    ):
        raise SentimentScoringError("sentiment backends disagree on text inventory")
    manifest = seal_manifest(
        {
            "schema_version": SUITE_MANIFEST_SCHEMA,
            "status": "complete",
            "immutable": True,
            "evaluation_id": EVALUATION_ID,
            "primary_backend": LEXICON_BACKEND,
            "robustness_backends": [DISTIL_BACKEND, FINBERT_BACKEND],
            "construct_nonpooling": {
                LEXICON_BACKEND: "monetary_policy_stance_hawkish_minus_dovish",
                DISTIL_BACKEND: "monetary_policy_stance_hawkish_minus_dovish",
                FINBERT_BACKEND: "financial_valence_positive_minus_negative",
                "cross_construct_composite_score": False,
            },
            "gated_unavailable_registry_only": {
                "repo_id": "gtfintechlab/FOMC-RoBERTa",
                "revision": "aa3bc4281fb1fe73c8872e09ad5c64b898f90d83",
                "role": "registered_but_not_an_acceptance_dependency",
                "availability": "manual_gated_not_locally_available",
                "used_for_scores": False,
            },
            "inventory_manifest": copy.deepcopy(inventory["manifest_binding"]),
            "backend_manifests": {
                backend_id: copy.deepcopy(backends[backend_id]["manifest_binding"])
                for backend_id in BACKEND_ORDER
            },
            "summaries": _summaries(backends),
            "implementation": _implementation_binding(),
        }
    )
    _write_readonly_json(output_path, manifest)
    return manifest


def load_and_validate_sentiment_suite(
    manifest_path: Path, *, deep_sources: bool = False
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    payload_sha = validate_manifest_integrity(manifest)
    if (
        manifest.get("schema_version") != SUITE_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("evaluation_id") != EVALUATION_ID
        or manifest.get("primary_backend") != LEXICON_BACKEND
        or manifest.get("robustness_backends") != [DISTIL_BACKEND, FINBERT_BACKEND]
    ):
        raise SentimentScoringError("sentiment suite header drift")
    _validate_implementation_binding(manifest.get("implementation"))
    inventory_binding = manifest.get("inventory_manifest")
    backend_bindings = manifest.get("backend_manifests")
    if not isinstance(inventory_binding, Mapping) or not isinstance(
        backend_bindings, Mapping
    ):
        raise SentimentScoringError("sentiment suite bindings missing")
    inventory = load_and_validate_inventory(
        Path(str(inventory_binding.get("path"))), deep_sources=deep_sources
    )
    if inventory_binding != inventory["manifest_binding"]:
        raise SentimentScoringError("sentiment suite inventory binding drift")
    backends: dict[str, Any] = {}
    for backend_id in BACKEND_ORDER:
        binding = backend_bindings.get(backend_id)
        if not isinstance(binding, Mapping):
            raise SentimentScoringError("sentiment suite backend binding missing")
        loaded = load_and_validate_meeting_scores(Path(str(binding.get("path"))))
        if binding != loaded["manifest_binding"]:
            raise SentimentScoringError("sentiment suite backend binding drift")
        backends[backend_id] = loaded
    if manifest.get("summaries") != _summaries(backends):
        raise SentimentScoringError("sentiment suite summaries drift")
    return {
        "manifest": manifest,
        "manifest_binding": _binding(manifest_path, payload_sha256=payload_sha),
        "inventory": inventory,
        "backends": backends,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument(
        "--suite-manifest", type=Path, default=DEFAULT_GENERATION_SUITE
    )
    prepare.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    prepare.add_argument("--cohort-sha256")
    prepare.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "preparation"
    )

    download = sub.add_parser("download-model")
    download.add_argument("--backend", choices=tuple(MODEL_CONTRACTS), required=True)
    download.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)

    lexicon = sub.add_parser("score-lexicon")
    lexicon.add_argument("--inventory-manifest", type=Path, required=True)
    lexicon.add_argument("--output-dir", type=Path, required=True)

    neural = sub.add_parser("score-neural")
    neural.add_argument("--backend", choices=tuple(MODEL_CONTRACTS), required=True)
    neural.add_argument("--inventory-manifest", type=Path, required=True)
    neural.add_argument("--model-manifest", type=Path, required=True)
    neural.add_argument("--output-dir", type=Path, required=True)
    neural.add_argument("--device", default="cuda:0")
    neural.add_argument("--batch-size", type=int, default=128)
    neural.add_argument("--expected-visible-gpu")

    seal = sub.add_parser("seal-suite")
    seal.add_argument("--inventory-manifest", type=Path, required=True)
    seal.add_argument("--lexicon-manifest", type=Path, required=True)
    seal.add_argument("--distil-manifest", type=Path, required=True)
    seal.add_argument("--finbert-manifest", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--deep-sources", action="store_true")

    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument(
        "--kind", choices=("inventory", "backend", "suite"), required=True
    )
    validate.add_argument("--deep-sources", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            cohort_sha256 = args.cohort_sha256 or sha256_file(args.cohort)
            value = prepare_inventory(
                suite_manifest=args.suite_manifest,
                cohort_path=args.cohort,
                cohort_sha256=cohort_sha256,
                output_dir=args.output_dir,
            )
        elif args.command == "download-model":
            value = download_and_seal_model(
                backend_id=args.backend, model_root=args.model_root
            )
        elif args.command == "score-lexicon":
            value = score_lexicon(
                inventory_manifest=args.inventory_manifest,
                output_dir=args.output_dir,
            )
        elif args.command == "score-neural":
            value = score_neural(
                backend_id=args.backend,
                inventory_manifest=args.inventory_manifest,
                model_manifest=args.model_manifest,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
                expected_visible_gpu=args.expected_visible_gpu,
            )
        elif args.command == "seal-suite":
            value = seal_sentiment_suite(
                inventory_manifest=args.inventory_manifest,
                backend_manifests={
                    LEXICON_BACKEND: args.lexicon_manifest,
                    DISTIL_BACKEND: args.distil_manifest,
                    FINBERT_BACKEND: args.finbert_manifest,
                },
                output_path=args.output,
                deep_sources=args.deep_sources,
            )
        elif args.kind == "inventory":
            value = load_and_validate_inventory(
                args.manifest, deep_sources=args.deep_sources
            )
        elif args.kind == "backend":
            value = load_and_validate_meeting_scores(args.manifest)
        else:
            value = load_and_validate_sentiment_suite(
                args.manifest, deep_sources=args.deep_sources
            )
    except SentimentScoringError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    output = value.get("manifest", value) if isinstance(value, Mapping) else value
    print(_canonical(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BACKEND_ORDER",
    "DISTIL_BACKEND",
    "FINBERT_BACKEND",
    "LEXICON_BACKEND",
    "MODEL_CONTRACTS",
    "SentimentScoringError",
    "_aggregate_meeting_rows",
    "_assemble_inventory_rows",
    "_lexicon_score",
    "_window_specs",
    "download_and_seal_model",
    "load_and_validate_inventory",
    "load_and_validate_meeting_scores",
    "load_and_validate_model_manifest",
    "load_and_validate_sentiment_suite",
    "prepare_inventory",
    "score_lexicon",
    "score_neural",
    "seal_sentiment_suite",
]
