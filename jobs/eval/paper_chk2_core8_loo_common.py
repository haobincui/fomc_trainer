"""Immutable preparation contract for the paper chk-2 Core8 LOO rerun.

The historical Core8 matrix is reused only as a frozen set of source texts,
interventions, references, and identities.  This module renders every prompt
again with the paper chk-2 system prompt and tokenizer, persists the exact
chat-template token IDs, proves the 1,024 neutral replacements remain
token-count matched, and records the task-specific post-training holdout audit.
It deliberately does not load a GPU model or generate any completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any

from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import (
    derive_row_seed,
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ID = "paper-chk2-cp50-core8-loo-n128-vllm-k10-v1"

SOURCE_RELEASE_ROOT = ROOT / (
    "dataset/processed/retrain_v2/"
    "chk3_minutes_external_holdout_1993_2008_all_regular_v1"
)
SOURCE_RELEASE_MANIFEST = SOURCE_RELEASE_ROOT / "release_manifest.json"
SOURCE_RELEASE_MANIFEST_SHA256 = (
    "82b045866d9ed6ccbc0d4f00014bffc97694859c2827215309dc8eabe5406937"
)
SOURCE_RELEASE_PAYLOAD_SHA256 = (
    "1bbbecdb4a5b71d4261b6a410dd0e76efa84e7386a2b85da86576c8609304a27"
)
SOURCE_PANEL = SOURCE_RELEASE_ROOT / "panels/core8.jsonl"
SOURCE_PANEL_SHA256 = (
    "eb74e0bb89614d089675cd0b1396f5963f590844fc94042759671e6170b5cec6"
)
FROZEN_INPUT_ROOT = ROOT / (
    "output/evaluation/main/"
    "chk3_cp318_core8_loo_full_n128_1993_2008_20260814_v1/inputs_v1"
)
FROZEN_INPUTS = FROZEN_INPUT_ROOT / "inputs.jsonl"
FROZEN_INPUTS_SHA256 = (
    "3b8ea039e2e20195f2adf7ab4994bc41cc8782df81f10d7e5bd53166a17b7b58"
)
LEGACY_SAMPLES_MANIFEST = FROZEN_INPUT_ROOT / "samples.json"
LEGACY_SAMPLES_MANIFEST_SHA256 = (
    "5ddaf97ef9ee7d957ca65a977d6527ea5cedc93e2b22fc15bc08627ea9e91efe"
)
LEGACY_SAMPLES_PAYLOAD_SHA256 = (
    "9b910c3eaa71926714d946efe8bfa28d410050f5b9e0cd92f21fdf7f29ee0478"
)

PAPER_RELEASE_ROOT = ROOT / (
    "dataset/processed/retrain_v2/"
    "chk2_chk1_final_analysis_synthetic_minutes_flash_official_reference_"
    "v6_downstream128_recovery_v1_20260831"
)
PAPER_RELEASE_MANIFEST = PAPER_RELEASE_ROOT / "release_manifest.json"
PAPER_RELEASE_MANIFEST_SHA256 = (
    "cb022dcd379e069e3d5ad54d7c7fdbc9e2ee29f958c85d5ff45d066f7728c097"
)
PROMPT_CONTRACT = PAPER_RELEASE_ROOT / "prompt_contract.json"
PROMPT_CONTRACT_SHA256 = (
    "1355f417b26cd7c44289951258e0ed86d1243448c64aef9d6bf33de7f4c5c616"
)
SYSTEM_PROMPT_SHA256 = (
    "4730a4ed585238547447ab850836db5a9fc67e5c3b1328b485c88d701ae78c4e"
)
USER_TEMPLATE_SHA256 = (
    "423e79849cb66d6361c986f705ea4d4b16a03e28fec5826113eb9c8030a976d0"
)
TRAINING_CONFIG = ROOT / (
    "configs/retrain_v2/"
    "paper_chk2_minutes_sft_chk1_cp200_v6_recovery_full3ep_lr1e6_20260901.yaml"
)
TRAINING_CONFIG_SHA256 = (
    "ff7ee316d41e990791fb9143b7367922a42fda3c9af7ffda5f95f275575a1a09"
)
PARENT_MODEL = ROOT / (
    "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
)
PARENT_MODEL_SHA256 = (
    "0989b94792f8ab6377e2979aaedeb37010b9a2c5e05c77a459b4e5cda5b806e3"
)
PARENT_MERGE_ATTESTATION = PARENT_MODEL / "merge_attestation.json"
PARENT_MERGE_ATTESTATION_SHA256 = (
    "bce783ee21582a5eadb291859e552e23e4897b32063765182f00d818e936abcf"
)
CHK2_ADAPTER = ROOT / (
    "output/training/retrain_v2/"
    "paper_chk2_chk1_cp200_minutes_v6_recovery_full3ep_lr1e6_v1_20260901/"
    "adapters/chk2/checkpoint-50"
)
SELECTION_RECEIPT = ROOT / (
    "output/evaluation/retrain_v2/"
    "paper_chk2_minutes_checkpoint_hard_gate_probe_v1_20260901/"
    "relaxed_checkpoint_selection_v1.json"
)
SELECTION_RECEIPT_SHA256 = (
    "9c2fcff135114ff1165c60b881f109c69bb3c0f3b04368157744e634b99e138f"
)

CHK2_RUNTIME_FILES: Mapping[str, str] = MappingProxyType(
    {
        "adapter_model.safetensors": (
            "158f73c538f6969c7d829f869ec79bff501caaa92c40bb61b564c957d92717bf"
        ),
        "adapter_config.json": (
            "1b92cf693cb4f3e2b727dafdd84967c8578438b3380626ec100053f810a398eb"
        ),
        "tokenizer.json": (
            "d91915040cfac999d8c55f4b5bc6e67367c065e3a7a4e4b9438ce1f256addd86"
        ),
        "tokenizer_config.json": (
            "1430c0e4827806c372aff86a80b40563deed50e76ae945aa1a84152cd31d063b"
        ),
        "special_tokens_map.json": (
            "c60e746a3f6de956bd5b2a60ed0222faff611ac95d9cb75d4ac79ca74a12a08b"
        ),
        "chat_template.jinja": (
            "56a1447ad31926fdc21fb07e56e5642bd9c850c4f52d8c8af7bbe5f079a84f5f"
        ),
    }
)
EXPECTED_TRANSFORMERS_VERSION = "4.57.6"
EXPECTED_TOKENIZERS_VERSION = "0.22.2"
EXPECTED_TOKENIZER_CLASS = (
    "transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast"
)

PARENT_CHK1_REPAIR_MANIFEST = ROOT / (
    "output/data/retrain_v2/chk1/"
    "chk1_reasoning_compressed_flash_max_v2_clean_20260809_candidate/"
    "audits/repair_manifest.jsonl"
)
PARENT_CHK1_REPAIR_MANIFEST_SHA256 = (
    "cd7dee86792c8190a5daceb99612c36a0396506f7dfa4da551a6f41cbb6c3d0b"
)
PAPER_SIDECAR_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "train": "51c82a70e5967a9b2463c733b4a38685d29fae8aeafd4264351a204966ca4f0c",
        "validation": "f61b70c3890408430954b7de6396c7960bc4c5379440d3ef5233be3d81e2036a",
        "test": "730360c9c66097c254243a344453a5ba1179da83d6cbbf4ca969958ade474ba3",
    }
)
PAPER_SPLIT_ROWS: Mapping[str, int] = MappingProxyType(
    {"train": 305, "validation": 42, "test": 44}
)
PRIOR_CORE8_SCORE_MANIFESTS: Mapping[str, tuple[Path, str]] = MappingProxyType(
    {
        "2026-08-14_greedy_k1": (
            ROOT
            / "output/evaluation/main/"
            "chk3_cp318_core8_loo_full_n128_1993_2008_20260814_v1/"
            "score_v1/manifest.json",
            "4226551fd40ea9b02678624756295d868929fecec8664352ff5282134dde8df8",
        ),
        "2026-08-17_stochastic_k5": (
            ROOT
            / "output/evaluation/main/"
            "chk3_cp318_core8_loo_vllm_k5_n128_1993_2008_20260817_v1/"
            "score_raw_semantic_dual_gpu_b10000_v2/manifest.json",
            "4df87c36a7215e86fc22c1a33b96b700344f523254c035e5efaf5d00383c5fee",
        ),
        "2026-08-24_stochastic_k10": (
            ROOT
            / "output/evaluation/main/"
            "chk3_cp318_core8_loo_vllm_k10_n128_1993_2008_20260824_v1/"
            "score_incremental_reuse_k5_raw_semantic_b10000_v1/manifest.json",
            "f21235c1386e12ead6ec2698d87a6b1746d80b6079e2fabc79ee584b581f7ef2",
        ),
    }
)

DEFAULT_OUTPUT_ROOT = ROOT / (
    "output/evaluation/retrain_v2/"
    "paper_chk2_cp50_core8_loo_vllm_k10_n128_1993_2008_"
    "t06_p095_b10000_v1_20260901"
)
PREPARATION_DIRNAME = "preparation"
MANIFEST_FILENAME = "evaluation_manifest.json"
LEDGER_FILENAME = "prompt_token_ledger.jsonl"
NEUTRAL_PROOFS_FILENAME = "neutral_token_match_proofs.jsonl"

PREPARATION_SCHEMA = "paper-chk2-core8-loo-preparation-manifest-v1"
LEDGER_ROW_SCHEMA = "paper-chk2-core8-loo-prompt-token-ledger-row-v1"
NEUTRAL_PROOF_SCHEMA = "paper-chk2-core8-loo-neutral-token-match-proof-v1"
HOLDOUT_AUDIT_SCHEMA = "paper-chk2-core8-loo-recorded-lineage-holdout-audit-v1"

TOPICS = (
    "Consumer-Price-Index-(CPI)",
    "GDP-Growth",
    "Government-Purchases",
    "Housing-Starts",
    "Industrial-Production",
    "Labour-Market",
    "Money-Supply",
    "Unemployment-Rate",
)
EXPECTED_MEETINGS = 128
VARIANTS_PER_MEETING = 17
EXPECTED_PROMPTS = EXPECTED_MEETINGS * VARIANTS_PER_MEETING
EXPECTED_NEUTRAL_PROOFS = EXPECTED_MEETINGS * len(TOPICS)
REPLICATE_SEEDS = (
    20260811,
    21260811,
    22260811,
    23260811,
    24260811,
    25260811,
    26260811,
    27260811,
    28260811,
    29260811,
)
EXPECTED_GENERATIONS = EXPECTED_PROMPTS * len(REPLICATE_SEEDS)
MAX_NEW_TOKENS = 2_560
MAX_MODEL_LEN = 4_096
MAX_PROMPT_TOKENS = MAX_MODEL_LEN - MAX_NEW_TOKENS
EXPECTED_MIN_PROMPT_TOKENS = 981
EXPECTED_MAX_PROMPT_TOKENS = 1_291
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 50
REPETITION_PENALTY = 1.0

LEDGER_KEYS = frozenset(
    {
        "schema_version",
        "absolute_prompt_index",
        "source_line_number",
        "sample_id",
        "meeting_id",
        "meeting_start_date",
        "meeting_rank",
        "arm",
        "intervention_topic",
        "variant_rank",
        "prompt_sha256",
        "source_analysis_sha256",
        "reference_minutes_sha256",
        "full_source_analysis_sha256",
        "full_reference_sha256",
        "messages_sha256",
        "prompt_token_count",
        "prompt_token_ids",
        "prompt_token_ids_sha256",
        "paired_seed_key",
        "row_sha256",
    }
)
NEUTRAL_PROOF_KEYS = frozenset(
    {
        "schema_version",
        "meeting_id",
        "intervention_topic",
        "full_sample_id",
        "neutral_sample_id",
        "full_absolute_prompt_index",
        "neutral_absolute_prompt_index",
        "full_prompt_token_count",
        "neutral_prompt_token_count",
        "token_count_equal",
        "full_prompt_token_ids_sha256",
        "neutral_prompt_token_ids_sha256",
        "replacement_sha256",
        "label_preserved",
        "numeric_surfaces_removed",
        "date_values_removed",
        "proof_sha256",
    }
)


class PaperChk2Core8PreparationError(RuntimeError):
    """A frozen binding, exact-token proof, or immutable artifact drifted."""


@dataclass(frozen=True, slots=True)
class FrozenSource:
    source_release_manifest: dict[str, Any]
    source_release_manifest_path: Path
    source_release_manifest_sha256: str
    panel_path: Path
    panel_sha256: str
    panel_rows: tuple[dict[str, Any], ...]
    inputs_path: Path
    inputs_sha256: str
    legacy_samples_manifest_path: Path
    legacy_samples_manifest_sha256: str
    legacy_samples_manifest: dict[str, Any]
    ordered_rows: tuple[dict[str, Any], ...]
    by_sample_id: Mapping[str, dict[str, Any]]
    full_sample_id_by_meeting: Mapping[str, str]
    meeting_ids: tuple[str, ...]
    topics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PaperChk2Bindings:
    parent_model_path: Path
    adapter_path: Path
    tokenizer_path: Path
    prompt_contract_path: Path
    prompt_contract: dict[str, Any]
    system_prompt: str
    user_template: str
    system_prompt_sha256: str
    user_template_sha256: str
    runtime_versions: Mapping[str, str]
    file_bindings: Mapping[str, dict[str, Any]]
    tokenizer: Any | None = None


@dataclass(frozen=True, slots=True)
class PreparedArtifacts:
    manifest: dict[str, Any]
    ledger_rows: tuple[dict[str, Any], ...]
    neutral_proofs: tuple[dict[str, Any], ...]
    preparation_dir: Path
    manifest_path: Path
    ledger_path: Path
    neutral_proofs_path: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperChk2Core8PreparationError(message)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise PaperChk2Core8PreparationError(
            f"value is not finite canonical JSON: {exc}"
        ) from exc


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    _require(
        resolved.is_file() and not resolved.is_symlink(),
        f"missing/non-regular JSON artifact: {resolved}",
    )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperChk2Core8PreparationError(f"invalid JSON {resolved}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON object required: {resolved}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    _require(
        resolved.is_file() and not resolved.is_symlink(),
        f"missing/non-regular JSONL artifact: {resolved}",
    )
    rows: list[dict[str, Any]] = []
    try:
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            for line_number, raw in enumerate(handle, 1):
                _require(raw.endswith("\n"), f"unterminated JSONL row: {resolved}:{line_number}")
                _require(raw.strip() != "", f"blank JSONL row: {resolved}:{line_number}")
                value = json.loads(raw)
                _require(isinstance(value, dict), f"JSONL object required: {resolved}:{line_number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperChk2Core8PreparationError(f"invalid JSONL {resolved}: {exc}") from exc
    return rows


def _file_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    _require(
        resolved.is_file() and not resolved.is_symlink(),
        f"cannot bind missing/non-regular file: {resolved}",
    )
    result: dict[str, Any] = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _verify_file(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    record = _file_record(path)
    _require(record["sha256"] == expected_sha256, f"{label} SHA-256 drift")
    return record


def _verify_record(record: Mapping[str, Any], *, rows: int | None = None) -> Path:
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    _require(path.is_file() and not path.is_symlink(), f"bound file missing: {path}")
    _require(path.stat().st_size == record.get("bytes"), f"bound file size drift: {path}")
    _require(sha256_file(path) == record.get("sha256"), f"bound file SHA drift: {path}")
    if rows is not None:
        _require(record.get("rows") == rows, f"bound row count drift: {path}")
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _render_user_prompt(user_template: str, source_analysis: str) -> str:
    marker = '"[SOURCE_ANALYSIS]"'
    _require(user_template.count(marker) == 1, "user template placeholder contract drift")
    return user_template.replace(marker, json.dumps(source_analysis, ensure_ascii=False))


def _expected_variant(rank: int) -> tuple[str, str | None]:
    if rank == 0:
        return "full", None
    topic = TOPICS[(rank - 1) // 2]
    return ("exact_deletion" if rank % 2 else "token_matched_neutral"), topic


def load_and_validate_frozen_source() -> FrozenSource:
    """Deep-validate and load the frozen 128 x 17 source matrix in file order."""

    _verify_file(
        SOURCE_RELEASE_MANIFEST,
        SOURCE_RELEASE_MANIFEST_SHA256,
        label="external release manifest",
    )
    release = _read_json(SOURCE_RELEASE_MANIFEST)
    try:
        payload_sha = validate_manifest_integrity(
            release, expected_payload_sha256=SOURCE_RELEASE_PAYLOAD_SHA256
        )
    except ValueError as exc:
        raise PaperChk2Core8PreparationError(
            f"external release manifest integrity failed: {exc}"
        ) from exc
    _require(payload_sha == SOURCE_RELEASE_PAYLOAD_SHA256, "external release payload drift")
    _require(
        release.get("schema_version") == "chk3-external-evaluation-release-v1"
        and release.get("status") == "passed"
        and release.get("immutable") is True
        and release.get("evaluation_only") is True
        and release.get("trainable") is False
        and release.get("checkpoint_selection_allowed") is False
        and release.get("meeting_count") == EXPECTED_MEETINGS
        and release.get("core8_rows") == EXPECTED_MEETINGS * len(TOPICS),
        "external release scope/count contract drift",
    )
    panel_record = (release.get("files") or {}).get("panels/core8.jsonl")
    _require(isinstance(panel_record, Mapping), "external Core8 panel binding missing")
    panel_file_record = _verify_file(SOURCE_PANEL, SOURCE_PANEL_SHA256, label="Core8 panel")
    _require(
        panel_record.get("path") == "panels/core8.jsonl"
        and panel_record.get("sha256") == SOURCE_PANEL_SHA256
        and panel_record.get("bytes") == panel_file_record["bytes"]
        and panel_record.get("rows") == EXPECTED_MEETINGS * len(TOPICS),
        "external release Core8 panel declaration drift",
    )
    panel_rows = _read_jsonl(SOURCE_PANEL)
    _require(len(panel_rows) == EXPECTED_MEETINGS * len(TOPICS), "Core8 panel is not N=1,024")
    panel_keys: set[tuple[str, str]] = set()
    panel_topics: Counter[str] = Counter()
    panel_meeting_dates: dict[str, str] = {}
    for row in panel_rows:
        meeting_id = str(row.get("meeting_id") or "")
        topic = str(row.get("topic") or "")
        start = str(row.get("meeting_start_date") or "")
        cutoff = str(row.get("evidence_cutoff") or "")
        _require(meeting_id and topic in TOPICS, "invalid Core8 meeting/topic identity")
        _require((meeting_id, topic) not in panel_keys, "duplicate Core8 meeting/topic")
        _require(
            cutoff == (date.fromisoformat(start) - timedelta(days=1)).isoformat(),
            f"Core8 D-1 cutoff drift: {meeting_id}:{topic}",
        )
        for text_key, hash_key in (
            ("source_analysis", "source_analysis_sha256"),
            ("reference_minutes", "reference_minutes_sha256"),
            ("prompt", "prompt_sha256"),
        ):
            text = row.get(text_key)
            _require(
                isinstance(text, str) and text and sha256_text(text) == row.get(hash_key),
                f"Core8 {text_key} hash drift: {meeting_id}:{topic}",
            )
        _require(
            row.get("official_minutes_role") == "secondary_context_not_model_input",
            "official Minutes role drift",
        )
        panel_keys.add((meeting_id, topic))
        panel_topics[topic] += 1
        previous = panel_meeting_dates.setdefault(meeting_id, start)
        _require(previous == start, f"Core8 meeting date drift: {meeting_id}")
    _require(set(panel_topics) == set(TOPICS), "Core8 topic inventory drift")
    _require(set(panel_topics.values()) == {EXPECTED_MEETINGS}, "Core8 topic closure drift")
    _require(len(panel_meeting_dates) == EXPECTED_MEETINGS, "Core8 meeting closure drift")

    _verify_file(
        LEGACY_SAMPLES_MANIFEST,
        LEGACY_SAMPLES_MANIFEST_SHA256,
        label="legacy source identity manifest",
    )
    legacy = _read_json(LEGACY_SAMPLES_MANIFEST)
    try:
        legacy_payload = validate_manifest_integrity(
            legacy, expected_payload_sha256=LEGACY_SAMPLES_PAYLOAD_SHA256
        )
    except ValueError as exc:
        raise PaperChk2Core8PreparationError(
            f"legacy source identity manifest integrity failed: {exc}"
        ) from exc
    _require(legacy_payload == LEGACY_SAMPLES_PAYLOAD_SHA256, "legacy source payload drift")
    _require(
        legacy.get("schema_version") == "chk3-core8-loo-full-samples-v1"
        and legacy.get("status") == "complete",
        "legacy source identity manifest status drift",
    )
    selection = legacy.get("selection")
    samples = legacy.get("samples")
    _require(
        isinstance(selection, Mapping)
        and selection.get("meetings") == EXPECTED_MEETINGS
        and selection.get("rows") == EXPECTED_PROMPTS
        and selection.get("variants_per_meeting") == VARIANTS_PER_MEETING
        and tuple(selection.get("topics") or ()) == TOPICS
        and isinstance(samples, list)
        and len(samples) == EXPECTED_PROMPTS,
        "legacy N=2,176 source-matrix declaration drift",
    )
    for key, expected in (
        ("source_release", SOURCE_RELEASE_MANIFEST_SHA256),
        ("source_panel", SOURCE_PANEL_SHA256),
        ("dataset", FROZEN_INPUTS_SHA256),
    ):
        raw = legacy.get(key)
        _require(isinstance(raw, Mapping) and raw.get("sha256") == expected, f"legacy {key} binding drift")
    _require(
        legacy.get("source_release_payload_sha256") == SOURCE_RELEASE_PAYLOAD_SHA256,
        "legacy external release payload binding drift",
    )

    _verify_file(FROZEN_INPUTS, FROZEN_INPUTS_SHA256, label="frozen 17-arm inputs")
    rows = _read_jsonl(FROZEN_INPUTS)
    _require(len(rows) == EXPECTED_PROMPTS, "frozen input matrix is not N=2,176")
    meeting_ids = tuple(str(value) for value in selection.get("meeting_ids") or ())
    _require(
        len(meeting_ids) == EXPECTED_MEETINGS
        and len(set(meeting_ids)) == EXPECTED_MEETINGS
        and set(meeting_ids) == set(panel_meeting_dates),
        "frozen meeting roster/order drift",
    )

    by_id: dict[str, dict[str, Any]] = {}
    full_ids: dict[str, str] = {}
    per_meeting_reference: dict[str, str] = {}
    per_meeting_full_analysis: dict[str, str] = {}
    neutral_proofs: list[dict[str, Any]] = []
    for index, (sample, row) in enumerate(zip(samples, rows, strict=True)):
        _require(isinstance(sample, Mapping), f"legacy sample is not an object: {index + 1}")
        meeting_rank, variant_rank = divmod(index, VARIANTS_PER_MEETING)
        meeting_id = meeting_ids[meeting_rank]
        expected_arm, expected_topic = _expected_variant(variant_rank)
        sample_id = str(row.get("sample_id") or "")
        _require(sample_id and sample_id not in by_id, f"duplicate/empty sample ID: {sample_id!r}")
        _require(sample.get("sample_id") == sample_id, f"sample identity/order drift: line {index + 1}")
        for key, expected in (
            ("meeting_id", meeting_id),
            ("meeting_rank", meeting_rank),
            ("variant_rank", variant_rank),
            ("arm", expected_arm),
            ("intervention_topic", expected_topic),
        ):
            _require(row.get(key) == expected and sample.get(key) == expected, f"matrix {key} drift: {sample_id}")
        start = panel_meeting_dates[meeting_id]
        _require(row.get("meeting_start_date") == start and sample.get("meeting_start_date") == start, f"meeting date drift: {sample_id}")
        for text_key, hash_key in (
            ("prompt", "prompt_sha256"),
            ("source_analysis", "source_analysis_sha256"),
            ("reference_minutes", "reference_minutes_sha256"),
        ):
            text = row.get(text_key)
            digest = row.get(hash_key)
            _require(isinstance(text, str) and text and sha256_text(text) == digest, f"{text_key} hash drift: {sample_id}")
            _require(sample.get(hash_key) == digest, f"legacy {hash_key} drift: {sample_id}")
        expected_prompt = (
            "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
            + canonical_json({"analysis": row["source_analysis"]})
        )
        _require(row["prompt"] == expected_prompt, f"frozen raw user prompt drift: {sample_id}")
        _require(row.get("full_source_analysis_sha256") == sample.get("full_source_analysis_sha256"), f"full-analysis binding drift: {sample_id}")
        _require(row.get("full_reference_sha256") == row.get("reference_minutes_sha256"), f"full-reference binding drift: {sample_id}")
        block_count = str(row["source_analysis"]).count("--- LOO-INDICATOR-BLOCK:")
        _require(block_count == (7 if expected_arm == "exact_deletion" else 8), f"indicator-block count drift: {sample_id}")
        if expected_arm == "full":
            _require(row.get("full_source_analysis_sha256") == row.get("source_analysis_sha256"), f"full analysis self-binding drift: {sample_id}")
            full_ids[meeting_id] = sample_id
            per_meeting_reference[meeting_id] = str(row["reference_minutes_sha256"])
            per_meeting_full_analysis[meeting_id] = str(row["source_analysis_sha256"])
        else:
            _require(
                row.get("reference_minutes_sha256") == per_meeting_reference.get(meeting_id)
                and row.get("full_source_analysis_sha256") == per_meeting_full_analysis.get(meeting_id),
                f"meeting-level target/full-analysis binding drift: {sample_id}",
            )
        proof = row.get("neutral_proof")
        if expected_arm == "token_matched_neutral":
            _require(isinstance(proof, Mapping), f"neutral proof missing: {sample_id}")
            _require(
                proof.get("meeting_id") == meeting_id
                and proof.get("topic") == expected_topic
                and proof.get("label_preserved") is True
                and proof.get("numeric_surfaces_removed") is True
                and proof.get("date_values_removed") is True
                and str(row["source_analysis"]).count("Analysis withheld.") == 1,
                f"neutral semantic proof drift: {sample_id}",
            )
            neutral_proofs.append(dict(proof))
        else:
            _require(proof is None, f"non-neutral row carries a neutral proof: {sample_id}")
        by_id[sample_id] = row

    _require(len(full_ids) == EXPECTED_MEETINGS, "one-Full-per-meeting closure drift")
    _require(len(neutral_proofs) == EXPECTED_NEUTRAL_PROOFS, "neutral proof closure drift")
    _require(
        legacy.get("neutral_replacement_proofs") == neutral_proofs,
        "legacy neutral proof inventory/order drift",
    )
    return FrozenSource(
        source_release_manifest=release,
        source_release_manifest_path=SOURCE_RELEASE_MANIFEST.resolve(),
        source_release_manifest_sha256=SOURCE_RELEASE_MANIFEST_SHA256,
        panel_path=SOURCE_PANEL.resolve(),
        panel_sha256=SOURCE_PANEL_SHA256,
        panel_rows=tuple(panel_rows),
        inputs_path=FROZEN_INPUTS.resolve(),
        inputs_sha256=FROZEN_INPUTS_SHA256,
        legacy_samples_manifest_path=LEGACY_SAMPLES_MANIFEST.resolve(),
        legacy_samples_manifest_sha256=LEGACY_SAMPLES_MANIFEST_SHA256,
        legacy_samples_manifest=legacy,
        ordered_rows=tuple(rows),
        by_sample_id=MappingProxyType(by_id),
        full_sample_id_by_meeting=MappingProxyType(full_ids),
        meeting_ids=meeting_ids,
        topics=TOPICS,
    )


def _runtime_file_bindings() -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for relative, expected_sha in CHK2_RUNTIME_FILES.items():
        bindings[f"chk2_adapter/{relative}"] = _verify_file(
            CHK2_ADAPTER / relative, expected_sha, label=f"chk2 cp50 {relative}"
        )
    return bindings


def verify_paper_chk2_bindings(*, load_tokenizer: bool = True) -> PaperChk2Bindings:
    """Verify cp50, its parent, prompt, tokenizer, and selection bindings."""

    paper_release_record = _verify_file(
        PAPER_RELEASE_MANIFEST,
        PAPER_RELEASE_MANIFEST_SHA256,
        label="paper chk2 release manifest",
    )
    paper_release = _read_json(PAPER_RELEASE_MANIFEST)
    _require(
        paper_release.get("schema_version") == "paper-chk2-downstream-recovery-release-v1"
        and paper_release.get("status") == "complete"
        and paper_release.get("immutable") is True
        and paper_release.get("training_only") is True
        and paper_release.get("source_rows") == 1743,
        "paper chk2 training-release scope drift",
    )
    parent_declared = paper_release.get("parent_checkpoint") or {}
    _require(
        parent_declared.get("model_sha256") == PARENT_MODEL_SHA256
        and parent_declared.get("model_path")
        == str(PARENT_MODEL.relative_to(ROOT)),
        "paper chk2 parent declaration drift",
    )
    prompt_record = _verify_file(PROMPT_CONTRACT, PROMPT_CONTRACT_SHA256, label="paper chk2 prompt contract")
    prompt = _read_json(PROMPT_CONTRACT)
    system_prompt = prompt.get("system_prompt")
    user_template = prompt.get("user_prompt_template")
    _require(
        prompt.get("schema_version") == "paper-chk2-student-prompt-contract-v1"
        and isinstance(system_prompt, str)
        and isinstance(user_template, str)
        and sha256_text(system_prompt) == SYSTEM_PROMPT_SHA256
        and sha256_text(user_template) == USER_TEMPLATE_SHA256
        and prompt.get("system_prompt_sha256") == SYSTEM_PROMPT_SHA256
        and prompt.get("user_prompt_template_sha256") == USER_TEMPLATE_SHA256,
        "paper chk2 prompt text/hash contract drift",
    )
    declared_prompt = paper_release.get("student_prompt_contract") or {}
    _require(
        declared_prompt.get("system_prompt_sha256") == SYSTEM_PROMPT_SHA256
        and declared_prompt.get("user_prompt_template_sha256") == USER_TEMPLATE_SHA256,
        "paper release prompt binding drift",
    )
    config_record = _verify_file(TRAINING_CONFIG, TRAINING_CONFIG_SHA256, label="paper chk2 training config")
    receipt_record = _verify_file(SELECTION_RECEIPT, SELECTION_RECEIPT_SHA256, label="paper chk2 cp50 selection receipt")
    receipt = _read_json(SELECTION_RECEIPT)
    scope = receipt.get("scope") or {}
    _require(
        receipt.get("status") == "selected"
        and receipt.get("selected_checkpoint") == 50
        and scope.get("selection_only") is True
        and scope.get("not_a_merge_or_promotion_authorization") is True
        and scope.get("evaluation_eligible") is False
        and scope.get("test_generation_performed") is False,
        "cp50 selection-only receipt drift",
    )
    merge_record = _verify_file(PARENT_MERGE_ATTESTATION, PARENT_MERGE_ATTESTATION_SHA256, label="chk1 merge attestation")
    parent_fingerprint = fingerprint_artifact_path(PARENT_MODEL)
    _require(parent_fingerprint.get("sha256") == PARENT_MODEL_SHA256, "chk1 parent model directory drift")

    adapter_config = _read_json(CHK2_ADAPTER / "adapter_config.json")
    _require(
        adapter_config.get("base_model_name_or_path") == str(PARENT_MODEL.relative_to(ROOT))
        and adapter_config.get("peft_type") == "LORA"
        and adapter_config.get("task_type") == "CAUSAL_LM",
        "cp50 adapter-to-parent binding drift",
    )
    file_bindings = _runtime_file_bindings()
    file_bindings.update(
        {
            "paper_release_manifest": paper_release_record,
            "prompt_contract": prompt_record,
            "training_config": config_record,
            "selection_receipt": receipt_record,
            "parent_merge_attestation": merge_record,
            "parent_model": parent_fingerprint,
        }
    )
    observed_versions = {
        "python": platform.python_version(),
        "transformers": str(_package_version("transformers")),
        "tokenizers": str(_package_version("tokenizers")),
    }
    tokenizer: Any | None = None
    if load_tokenizer:
        _require(
            observed_versions["transformers"] == EXPECTED_TRANSFORMERS_VERSION
            and observed_versions["tokenizers"] == EXPECTED_TOKENIZERS_VERSION,
            "tokenizer runtime version drift; preparation requires "
            f"transformers=={EXPECTED_TRANSFORMERS_VERSION} and "
            f"tokenizers=={EXPECTED_TOKENIZERS_VERSION}, observed "
            f"{observed_versions['transformers']}/{observed_versions['tokenizers']}",
        )
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(CHK2_ADAPTER),
                local_files_only=True,
                trust_remote_code=True,
                use_fast=True,
                fix_mistral_regex=False,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise PaperChk2Core8PreparationError(f"cannot load pinned cp50 tokenizer: {exc}") from exc
        class_name = f"{type(tokenizer).__module__}.{type(tokenizer).__name__}"
        _require(class_name == EXPECTED_TOKENIZER_CLASS, f"tokenizer class drift: {class_name}")
        _require(
            isinstance(tokenizer.chat_template, str)
            and sha256_text(tokenizer.chat_template) == CHK2_RUNTIME_FILES["chat_template.jinja"],
            "loaded tokenizer chat-template drift",
        )
    return PaperChk2Bindings(
        parent_model_path=PARENT_MODEL.resolve(),
        adapter_path=CHK2_ADAPTER.resolve(),
        tokenizer_path=CHK2_ADAPTER.resolve(),
        prompt_contract_path=PROMPT_CONTRACT.resolve(),
        prompt_contract=prompt,
        system_prompt=system_prompt,
        user_template=user_template,
        system_prompt_sha256=SYSTEM_PROMPT_SHA256,
        user_template_sha256=USER_TEMPLATE_SHA256,
        runtime_versions=MappingProxyType(observed_versions),
        file_bindings=MappingProxyType(file_bindings),
        tokenizer=tokenizer,
    )


def _prompt_token_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    try:
        values = tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True
        )
    except (TypeError, ValueError) as exc:
        raise PaperChk2Core8PreparationError(f"chat-template rendering failed: {exc}") from exc
    if hasattr(values, "tolist"):
        values = values.tolist()
    _require(isinstance(values, list) and values, "chat template returned no token IDs")
    _require(not isinstance(values[0], list), "chat template returned batched token IDs")
    token_ids = [int(value) for value in values]
    _require(all(value >= 0 for value in token_ids), "chat template returned a negative token ID")
    return token_ids


def _row_digest(row: Mapping[str, Any], digest_key: str) -> str:
    payload = dict(row)
    payload.pop(digest_key, None)
    return sha256_text(canonical_json(payload))


def build_exact_token_ledger(
    source: FrozenSource,
    bindings: PaperChk2Bindings,
    *,
    tokenizer: Any | None = None,
) -> tuple[dict[str, Any], ...]:
    """Render and materialize all 2,176 paper-chk2 exact prompt token IDs."""

    active_tokenizer = tokenizer if tokenizer is not None else bindings.tokenizer
    _require(active_tokenizer is not None, "a loaded tokenizer is required to build the ledger")
    _require(len(source.ordered_rows) == EXPECTED_PROMPTS, "source ledger coverage is not N=2,176")
    ledger: list[dict[str, Any]] = []
    for absolute_prompt_index, row in enumerate(source.ordered_rows):
        sample_id = str(row["sample_id"])
        expected_prompt = _render_user_prompt(bindings.user_template, str(row["source_analysis"]))
        _require(expected_prompt == row["prompt"], f"new user-template rendering drift: {sample_id}")
        messages = [
            {"role": "system", "content": bindings.system_prompt},
            {"role": "user", "content": expected_prompt},
        ]
        token_ids = _prompt_token_ids(active_tokenizer, messages)
        _require(
            len(token_ids) <= MAX_PROMPT_TOKENS,
            f"prompt exceeds {MAX_PROMPT_TOKENS} tokens: {sample_id} ({len(token_ids)})",
        )
        paired_seed_key = source.full_sample_id_by_meeting[str(row["meeting_id"])]
        output: dict[str, Any] = {
            "schema_version": LEDGER_ROW_SCHEMA,
            "absolute_prompt_index": absolute_prompt_index,
            "source_line_number": absolute_prompt_index + 1,
            "sample_id": sample_id,
            "meeting_id": row["meeting_id"],
            "meeting_start_date": row["meeting_start_date"],
            "meeting_rank": row["meeting_rank"],
            "arm": row["arm"],
            "intervention_topic": row["intervention_topic"],
            "variant_rank": row["variant_rank"],
            "prompt_sha256": row["prompt_sha256"],
            "source_analysis_sha256": row["source_analysis_sha256"],
            "reference_minutes_sha256": row["reference_minutes_sha256"],
            "full_source_analysis_sha256": row["full_source_analysis_sha256"],
            "full_reference_sha256": row["full_reference_sha256"],
            "messages_sha256": sha256_text(canonical_json(messages)),
            "prompt_token_count": len(token_ids),
            "prompt_token_ids": token_ids,
            "prompt_token_ids_sha256": sha256_text(canonical_json(token_ids)),
            "paired_seed_key": paired_seed_key,
        }
        output["row_sha256"] = _row_digest(output, "row_sha256")
        _require(set(output) == LEDGER_KEYS, "internal ledger key inventory drift")
        ledger.append(output)
    observed_counts = [int(row["prompt_token_count"]) for row in ledger]
    _require(
        min(observed_counts) == EXPECTED_MIN_PROMPT_TOKENS
        and max(observed_counts) == EXPECTED_MAX_PROMPT_TOKENS,
        "global prompt-token range drift; expected "
        f"{EXPECTED_MIN_PROMPT_TOKENS}--{EXPECTED_MAX_PROMPT_TOKENS}, observed "
        f"{min(observed_counts)}--{max(observed_counts)}",
    )
    return tuple(ledger)


def build_neutral_token_match_proofs(
    source: FrozenSource,
    ledger_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Seal token-count and legacy semantic proofs for all neutral arms."""

    _require(len(ledger_rows) == EXPECTED_PROMPTS, "ledger proof input is not N=2,176")
    by_id = {str(row.get("sample_id")): row for row in ledger_rows}
    _require(len(by_id) == EXPECTED_PROMPTS, "ledger proof input has duplicate IDs")
    proofs: list[dict[str, Any]] = []
    for source_row in source.ordered_rows:
        if source_row.get("arm") != "token_matched_neutral":
            continue
        neutral = by_id[str(source_row["sample_id"])]
        full_id = source.full_sample_id_by_meeting[str(source_row["meeting_id"])]
        full = by_id[full_id]
        legacy = source_row.get("neutral_proof")
        _require(isinstance(legacy, Mapping), f"neutral semantic proof missing: {source_row['sample_id']}")
        equal = neutral.get("prompt_token_count") == full.get("prompt_token_count")
        _require(equal, f"new prompt neutral token-count mismatch: {source_row['sample_id']}")
        proof: dict[str, Any] = {
            "schema_version": NEUTRAL_PROOF_SCHEMA,
            "meeting_id": source_row["meeting_id"],
            "intervention_topic": source_row["intervention_topic"],
            "full_sample_id": full_id,
            "neutral_sample_id": source_row["sample_id"],
            "full_absolute_prompt_index": full["absolute_prompt_index"],
            "neutral_absolute_prompt_index": neutral["absolute_prompt_index"],
            "full_prompt_token_count": full["prompt_token_count"],
            "neutral_prompt_token_count": neutral["prompt_token_count"],
            "token_count_equal": True,
            "full_prompt_token_ids_sha256": full["prompt_token_ids_sha256"],
            "neutral_prompt_token_ids_sha256": neutral["prompt_token_ids_sha256"],
            "replacement_sha256": legacy["replacement_sha256"],
            "label_preserved": legacy["label_preserved"],
            "numeric_surfaces_removed": legacy["numeric_surfaces_removed"],
            "date_values_removed": legacy["date_values_removed"],
        }
        proof["proof_sha256"] = _row_digest(proof, "proof_sha256")
        _require(set(proof) == NEUTRAL_PROOF_KEYS, "internal neutral-proof key inventory drift")
        proofs.append(proof)
    _require(len(proofs) == EXPECTED_NEUTRAL_PROOFS, "neutral proof output is not N=1,024")
    return tuple(proofs)


def build_recorded_lineage_holdout_audit(source: FrozenSource) -> dict[str, Any]:
    """Audit recorded task-specific post-training lineage against the holdout."""

    parent_record = _verify_file(
        PARENT_CHK1_REPAIR_MANIFEST,
        PARENT_CHK1_REPAIR_MANIFEST_SHA256,
        label="parent chk1 repair manifest",
    )
    parent_rows = _read_jsonl(PARENT_CHK1_REPAIR_MANIFEST)
    _require(len(parent_rows) == 1743, "parent chk1 recorded population is not N=1,743")
    date_pattern = re.compile(r"chk1-analysis-(\d{4}-\d{2}-\d{2})-")
    parent_dates: set[str] = set()
    parent_source_hashes: set[str] = set()
    parent_answer_hashes: set[str] = set()
    for row in parent_rows:
        match = date_pattern.search(str(row.get("sample_id") or ""))
        _require(match is not None, "parent chk1 sample ID has no meeting date")
        parent_dates.add(match.group(1))
        for key in (
            "provided_data_sha256",
            "candidate_answer_sha256",
            "source_answer_sha256",
            "prompt_sha256",
        ):
            value = row.get(key)
            _require(isinstance(value, str) and len(value) == 64, f"parent chk1 {key} drift")
            parent_source_hashes.add(value)
        parent_answer_hashes.add(str(row["candidate_answer_sha256"]))

    paper_release_record = _verify_file(
        PAPER_RELEASE_MANIFEST,
        PAPER_RELEASE_MANIFEST_SHA256,
        label="paper chk2 release manifest",
    )
    paper_release = _read_json(PAPER_RELEASE_MANIFEST)
    paper_rows: list[dict[str, Any]] = []
    sidecar_records: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        relative = Path("minutes_alignment/manifests") / f"{split}.jsonl"
        path = PAPER_RELEASE_ROOT / relative
        record = _verify_file(path, PAPER_SIDECAR_SHA256[split], label=f"paper chk2 {split} sidecar")
        rows = _read_jsonl(path)
        _require(len(rows) == PAPER_SPLIT_ROWS[split], f"paper chk2 {split} sidecar count drift")
        declared = paper_release["artifacts"]["minutes_alignment"][split]["manifest"]
        _require(
            declared.get("path") == relative.as_posix()
            and declared.get("sha256") == PAPER_SIDECAR_SHA256[split]
            and declared.get("rows") == PAPER_SPLIT_ROWS[split]
            and declared.get("bytes") == record["bytes"],
            f"paper chk2 {split} release binding drift",
        )
        for row in rows:
            _require(row.get("split") == split, f"paper chk2 {split} row split drift")
        record["rows"] = len(rows)
        sidecar_records[split] = record
        paper_rows.extend(rows)
    _require(len(paper_rows) == 391, "paper chk2 recorded population is not N=391")
    paper_dates = {str(row.get("meeting_date") or "") for row in paper_rows}
    paper_hashes = {str(row.get("source_analysis_sha256") or "") for row in paper_rows}
    _require(len(paper_dates) == 124 and "" not in paper_dates, "paper chk2 meeting closure drift")
    _require(len(paper_hashes) == 391 and "" not in paper_hashes, "paper chk2 source hash closure drift")

    external_dates = {str(row["meeting_start_date"]) for row in source.ordered_rows}
    external_hashes = {str(row["source_analysis_sha256"]) for row in source.ordered_rows}
    external_hashes.update(str(row["source_analysis_sha256"]) for row in source.panel_rows)
    external_parent_date_overlap = external_dates & parent_dates
    external_paper_date_overlap = external_dates & paper_dates
    external_parent_hash_overlap = external_hashes & parent_source_hashes
    external_paper_hash_overlap = external_hashes & paper_hashes
    _require(not external_parent_date_overlap, "external dates overlap recorded parent chk1 dates")
    _require(not external_paper_date_overlap, "external dates overlap recorded paper chk2 dates")
    _require(not external_parent_hash_overlap, "external source hashes overlap recorded parent chk1 hashes")
    _require(not external_paper_hash_overlap, "external source hashes overlap recorded paper chk2 hashes")
    _require(max(external_dates) < min(parent_dates) == min(paper_dates), "external/training temporal separation drift")
    _require(paper_hashes <= parent_answer_hashes, "paper chk2 sources no longer bind to parent chk1 answers")

    prior_evidence: dict[str, dict[str, Any]] = {}
    for label, (path, expected_sha) in PRIOR_CORE8_SCORE_MANIFESTS.items():
        prior_evidence[label] = _verify_file(path, expected_sha, label=f"prior Core8 score {label}")
    return {
        "schema_version": HOLDOUT_AUDIT_SCHEMA,
        "claim_scope": "held_out_from_recorded_task_specific_post_training_and_checkpoint_selection_only",
        "status": "passed_with_required_caveats",
        "bindings": {
            "parent_chk1_repair_manifest": {**parent_record, "rows": len(parent_rows)},
            "paper_chk2_release_manifest": paper_release_record,
            "paper_chk2_sidecars": sidecar_records,
            "prior_core8_score_manifests": prior_evidence,
        },
        "populations": {
            "external": {
                "meeting_dates": len(external_dates),
                "minimum_meeting_date": min(external_dates),
                "maximum_meeting_date": max(external_dates),
                "loo_input_rows": len(source.ordered_rows),
                "panel_rows": len(source.panel_rows),
                "unique_source_hashes_across_loo_inputs_and_panel": len(external_hashes),
            },
            "parent_chk1": {
                "rows": len(parent_rows),
                "meeting_dates": len(parent_dates),
                "minimum_meeting_date": min(parent_dates),
                "maximum_meeting_date": max(parent_dates),
                "source_like_hash_inventory_fields": [
                    "provided_data_sha256",
                    "candidate_answer_sha256",
                    "source_answer_sha256",
                    "prompt_sha256",
                ],
                "unique_source_like_hashes": len(parent_source_hashes),
            },
            "paper_chk2": {
                "rows": len(paper_rows),
                "meeting_dates": len(paper_dates),
                "minimum_meeting_date": min(paper_dates),
                "maximum_meeting_date": max(paper_dates),
                "unique_source_analysis_hashes": len(paper_hashes),
                "sources_bound_to_parent_chk1_answers": len(paper_hashes & parent_answer_hashes),
            },
        },
        "tests": {
            "external_max_date_precedes_parent_chk1_min_date": True,
            "external_max_date_precedes_paper_chk2_min_date": True,
            "external_parent_chk1_meeting_date_overlap": 0,
            "external_paper_chk2_meeting_date_overlap": 0,
            "external_parent_chk1_source_hash_overlap": 0,
            "external_paper_chk2_source_hash_overlap": 0,
        },
        "required_caveats": [
            "The same frozen Core8 panel was scored in recorded experiments on 2026-08-14, 2026-08-17, and 2026-08-24; repeated-holdout and model-development reuse cannot be ruled out.",
            "Independence from base-model pretraining is not claimed.",
            "This historical holdout is not a prospective evaluation.",
            "Adjacent-vintage macroeconomic observations may overlap even though exact meeting dates and recorded source hashes do not.",
            "The defensible leakage statement is limited to recorded task-specific post-training and checkpoint selection.",
        ],
    }


def canonical_case_coordinates(absolute_case_index: int) -> dict[str, int]:
    """Map the sample-major K=10 generation index to canonical coordinates."""

    _require(
        isinstance(absolute_case_index, int)
        and not isinstance(absolute_case_index, bool)
        and 0 <= absolute_case_index < EXPECTED_GENERATIONS,
        "absolute case index is out of range",
    )
    absolute_prompt_index, replicate_id = divmod(absolute_case_index, len(REPLICATE_SEEDS))
    meeting_index, variant_rank = divmod(absolute_prompt_index, VARIANTS_PER_MEETING)
    return {
        "absolute_case_index": absolute_case_index,
        "absolute_prompt_index": absolute_prompt_index,
        "replicate_id": replicate_id,
        "replicate_seed": REPLICATE_SEEDS[replicate_id],
        "meeting_index": meeting_index,
        "variant_rank": variant_rank,
        "paired_block_index": meeting_index * len(REPLICATE_SEEDS) + replicate_id,
    }


def row_seed_for_case(ledger_row: Mapping[str, Any], replicate_id: int) -> int:
    """Return the paired common-random-number seed for one ledger row."""

    _require(
        isinstance(replicate_id, int)
        and not isinstance(replicate_id, bool)
        and 0 <= replicate_id < len(REPLICATE_SEEDS),
        "replicate ID is out of range",
    )
    paired_seed_key = str(ledger_row.get("paired_seed_key") or "")
    _require(paired_seed_key != "", "ledger row has no paired seed key")
    return derive_row_seed(REPLICATE_SEEDS[replicate_id], paired_seed_key)


def _write_new_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_new_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_new_bytes(
        path,
        "".join(canonical_json(dict(row)) + "\n" for row in rows).encode("utf-8"),
    )


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    _write_new_bytes(path, payload)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_paths(output_root: Path) -> tuple[Path, Path, Path, Path]:
    preparation = output_root.expanduser().resolve() / PREPARATION_DIRNAME
    return (
        preparation,
        preparation / MANIFEST_FILENAME,
        preparation / LEDGER_FILENAME,
        preparation / NEUTRAL_PROOFS_FILENAME,
    )


def build_preparation_artifacts(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> PreparedArtifacts:
    """Create the write-once preparation directory, or verify it if it exists."""

    unresolved = output_root.expanduser()
    _require(not unresolved.is_symlink(), "refusing a symlink output root")
    output_root = unresolved.resolve()
    preparation, _manifest_path, _ledger_path, _proof_path = _artifact_paths(output_root)
    if preparation.exists() or preparation.is_symlink():
        return validate_preparation_artifacts(output_root, verify_static_bindings=True)

    source = load_and_validate_frozen_source()
    bindings = verify_paper_chk2_bindings(load_tokenizer=True)
    ledger = build_exact_token_ledger(source, bindings)
    proofs = build_neutral_token_match_proofs(source, ledger)
    holdout_audit = build_recorded_lineage_holdout_audit(source)
    prompt_counts = [int(row["prompt_token_count"]) for row in ledger]

    output_root.mkdir(parents=True, exist_ok=True)
    _require(not output_root.is_symlink(), "output root became a symlink")
    temporary = Path(tempfile.mkdtemp(prefix=".paper_chk2_core8_preparation.", dir=output_root))
    temp_ledger = temporary / LEDGER_FILENAME
    temp_proofs = temporary / NEUTRAL_PROOFS_FILENAME
    temp_manifest = temporary / MANIFEST_FILENAME
    published = False
    try:
        _write_new_jsonl(temp_ledger, ledger)
        _write_new_jsonl(temp_proofs, proofs)
        ledger_record = {
            **_file_record(temp_ledger, rows=EXPECTED_PROMPTS),
            "path": str((preparation / LEDGER_FILENAME).resolve()),
            "row_schema_version": LEDGER_ROW_SCHEMA,
        }
        proof_record = {
            **_file_record(temp_proofs, rows=EXPECTED_NEUTRAL_PROOFS),
            "path": str((preparation / NEUTRAL_PROOFS_FILENAME).resolve()),
            "row_schema_version": NEUTRAL_PROOF_SCHEMA,
        }
        manifest = seal_manifest(
            {
                "schema_version": PREPARATION_SCHEMA,
                "evaluation_id": EVALUATION_ID,
                "created_at_utc": _utc_now(),
                "status": "complete",
                "sealed": True,
                "immutable": True,
                "bindings": {
                    "external_release_manifest": _file_record(SOURCE_RELEASE_MANIFEST),
                    "external_release_payload_sha256": SOURCE_RELEASE_PAYLOAD_SHA256,
                    "core8_panel": _file_record(SOURCE_PANEL, rows=len(source.panel_rows)),
                    "frozen_17_arm_inputs": _file_record(FROZEN_INPUTS, rows=len(source.ordered_rows)),
                    "legacy_identity_order_manifest": {
                        **_file_record(LEGACY_SAMPLES_MANIFEST),
                        "payload_sha256": LEGACY_SAMPLES_PAYLOAD_SHA256,
                        "role": "source_identity_order_and_legacy_neutral_semantic_proofs_only",
                    },
                    "paper_chk2": {key: dict(value) for key, value in bindings.file_bindings.items()},
                    "prompt": {
                        "system_prompt_sha256": bindings.system_prompt_sha256,
                        "user_template_sha256": bindings.user_template_sha256,
                        "opening_think_supplied_by_chat_template": True,
                        "response_boundary": "</think>",
                    },
                    "tokenizer_runtime": {
                        "required_transformers": EXPECTED_TRANSFORMERS_VERSION,
                        "required_tokenizers": EXPECTED_TOKENIZERS_VERSION,
                        "observed": dict(bindings.runtime_versions),
                        "loader": "AutoTokenizer.from_pretrained",
                        "loader_kwargs": {
                            "local_files_only": True,
                            "trust_remote_code": True,
                            "use_fast": True,
                            "fix_mistral_regex": False,
                        },
                        "tokenizer_class": EXPECTED_TOKENIZER_CLASS,
                    },
                },
                "prepared_files": {
                    "prompt_token_ledger": ledger_record,
                    "neutral_token_match_proofs": proof_record,
                },
                "coverage": {
                    "meetings": EXPECTED_MEETINGS,
                    "topics": len(TOPICS),
                    "prompts": EXPECTED_PROMPTS,
                    "arms": {
                        "full": EXPECTED_MEETINGS,
                        "exact_deletion": EXPECTED_NEUTRAL_PROOFS,
                        "token_matched_neutral": EXPECTED_NEUTRAL_PROOFS,
                    },
                    "variants_per_meeting": VARIANTS_PER_MEETING,
                    "replicates": len(REPLICATE_SEEDS),
                    "generation_rows": EXPECTED_GENERATIONS,
                },
                "matrix_contract": {
                    "order": "frozen_inputs_jsonl_chronological_meeting_rank_then_variant_rank_0_to_16",
                    "sample_id": "core8-loo::<meeting_id>::<arm>::<topic-or-none>",
                    "full_variant_rank": 0,
                    "topic_variant_order": "for_each_topic_exact_deletion_then_token_matched_neutral",
                    "topics": list(TOPICS),
                    "reference": "one_fixed_synthetic_full_Core8_reference_per_meeting_shared_across_17_arms",
                },
                "prompt_budget": {
                    "minimum_prompt_tokens": min(prompt_counts),
                    "maximum_prompt_tokens": max(prompt_counts),
                    "max_prompt_tokens": MAX_PROMPT_TOKENS,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "max_model_len": MAX_MODEL_LEN,
                    "over_budget_rows": 0,
                    "neutral_token_count_mismatches": 0,
                },
                "generation_design": {
                    "backend": "vllm_bfloat16_no_quantization",
                    "model": "paper_chk2_chk1_cp200_plus_cp50_lora_adapter",
                    "replicate_seeds": list(REPLICATE_SEEDS),
                    "replicates": len(REPLICATE_SEEDS),
                    "rows": EXPECTED_GENERATIONS,
                    "canonical_case_order": "absolute_prompt_index_then_replicate_id",
                    "absolute_prompt_index": "absolute_case_index//10",
                    "replicate_id": "absolute_case_index%10",
                    "row_seed": "derive_row_seed(replicate_seed,meeting_full_sample_id)",
                    "common_random_numbers": "same_meeting_replicate_across_all_17_arms",
                    "prompt_input": "persisted_exact_token_ids_no_runtime_chat_templating",
                    "fresh_generation_required": True,
                    "historical_outputs_or_scores_reused": False,
                    "decoding": {
                        "do_sample": True,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "top_k": TOP_K,
                        "repetition_penalty": REPETITION_PENALTY,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "max_model_len": MAX_MODEL_LEN,
                        "dtype": "bfloat16",
                        "quantization": None,
                    },
                },
                "recorded_lineage_holdout_audit": holdout_audit,
                "interpretation": {
                    "estimand": "target_relative_multi_block_prompt_sensitivity",
                    "not_causal_topic_importance": True,
                    "reference_is_official_minutes": False,
                    "reference_description": "fixed synthetic concatenated full-Core8 Minutes-style target",
                },
                "implementation": _file_record(Path(__file__).resolve()),
                "mutation_policy": "write_once_preparation_directory_existing_directory_verify_only",
            }
        )
        validate_manifest_integrity(manifest)
        _write_new_json(temp_manifest, manifest)
        _fsync_directory(temporary)
        try:
            os.rename(temporary, preparation)
        except OSError as exc:
            if preparation.exists():
                return validate_preparation_artifacts(output_root, verify_static_bindings=True)
            raise PaperChk2Core8PreparationError(f"cannot publish preparation directory: {exc}") from exc
        published = True
        _fsync_directory(output_root)
        return validate_preparation_artifacts(output_root, verify_static_bindings=False)
    finally:
        if not published and temporary.exists():
            for candidate in (temp_manifest, temp_proofs, temp_ledger):
                try:
                    candidate.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                temporary.rmdir()
            except OSError:
                pass


def _validate_ledger_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(len(rows) == EXPECTED_PROMPTS, "prepared ledger is not N=2,176")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        _require(set(row) == LEDGER_KEYS, f"ledger key inventory drift: line {index + 1}")
        _require(row.get("schema_version") == LEDGER_ROW_SCHEMA, f"ledger schema drift: line {index + 1}")
        _require(row.get("absolute_prompt_index") == index and row.get("source_line_number") == index + 1, f"ledger order drift: line {index + 1}")
        sample_id = str(row.get("sample_id") or "")
        _require(sample_id and sample_id not in seen, f"ledger duplicate/empty sample ID: {sample_id!r}")
        seen.add(sample_id)
        ids = row.get("prompt_token_ids")
        _require(
            isinstance(ids, list)
            and ids
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in ids),
            f"invalid ledger token IDs: {sample_id}",
        )
        _require(row.get("prompt_token_count") == len(ids) <= MAX_PROMPT_TOKENS, f"ledger prompt count/budget drift: {sample_id}")
        _require(row.get("prompt_token_ids_sha256") == sha256_text(canonical_json(ids)), f"ledger token-ID hash drift: {sample_id}")
        _require(row.get("row_sha256") == _row_digest(row, "row_sha256"), f"ledger row digest drift: {sample_id}")
    observed_counts = [int(row["prompt_token_count"]) for row in rows]
    _require(
        min(observed_counts) == EXPECTED_MIN_PROMPT_TOKENS
        and max(observed_counts) == EXPECTED_MAX_PROMPT_TOKENS,
        "prepared global prompt-token range drift",
    )


def _validate_neutral_proofs(proofs: Sequence[Mapping[str, Any]]) -> None:
    _require(len(proofs) == EXPECTED_NEUTRAL_PROOFS, "prepared neutral proofs are not N=1,024")
    seen: set[str] = set()
    for index, proof in enumerate(proofs):
        _require(set(proof) == NEUTRAL_PROOF_KEYS, f"neutral-proof key drift: line {index + 1}")
        sample_id = str(proof.get("neutral_sample_id") or "")
        _require(sample_id and sample_id not in seen, f"neutral-proof duplicate/empty ID: {sample_id!r}")
        seen.add(sample_id)
        _require(
            proof.get("schema_version") == NEUTRAL_PROOF_SCHEMA
            and proof.get("token_count_equal") is True
            and proof.get("full_prompt_token_count") == proof.get("neutral_prompt_token_count")
            and proof.get("label_preserved") is True
            and proof.get("numeric_surfaces_removed") is True
            and proof.get("date_values_removed") is True,
            f"neutral proof contract drift: {sample_id}",
        )
        _require(proof.get("proof_sha256") == _row_digest(proof, "proof_sha256"), f"neutral proof digest drift: {sample_id}")


def validate_preparation_artifacts(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    verify_static_bindings: bool = True,
) -> PreparedArtifacts:
    """Validate the sealed preparation and optionally re-audit every source."""

    preparation, manifest_path, ledger_path, proof_path = _artifact_paths(output_root)
    _require(preparation.is_dir() and not preparation.is_symlink(), f"preparation directory missing: {preparation}")
    manifest = _read_json(manifest_path)
    try:
        validate_manifest_integrity(manifest)
    except ValueError as exc:
        raise PaperChk2Core8PreparationError(f"preparation manifest integrity failed: {exc}") from exc
    _require(
        manifest.get("schema_version") == PREPARATION_SCHEMA
        and manifest.get("evaluation_id") == EVALUATION_ID
        and manifest.get("status") == "complete"
        and manifest.get("sealed") is True
        and manifest.get("immutable") is True,
        "preparation manifest status/schema drift",
    )
    files = manifest.get("prepared_files") or {}
    ledger_record = files.get("prompt_token_ledger")
    proof_record = files.get("neutral_token_match_proofs")
    _require(isinstance(ledger_record, Mapping) and isinstance(proof_record, Mapping), "prepared file bindings missing")
    _require(Path(str(ledger_record.get("path"))).resolve() == ledger_path, "ledger path binding drift")
    _require(Path(str(proof_record.get("path"))).resolve() == proof_path, "neutral-proof path binding drift")
    _verify_record(ledger_record, rows=EXPECTED_PROMPTS)
    _verify_record(proof_record, rows=EXPECTED_NEUTRAL_PROOFS)
    ledger = _read_jsonl(ledger_path)
    proofs = _read_jsonl(proof_path)
    _validate_ledger_rows(ledger)
    _validate_neutral_proofs(proofs)
    prompt_budget = manifest.get("prompt_budget") or {}
    _require(
        prompt_budget.get("minimum_prompt_tokens") == EXPECTED_MIN_PROMPT_TOKENS
        and prompt_budget.get("maximum_prompt_tokens") == EXPECTED_MAX_PROMPT_TOKENS
        and prompt_budget.get("max_prompt_tokens") == MAX_PROMPT_TOKENS
        and prompt_budget.get("over_budget_rows") == 0
        and prompt_budget.get("neutral_token_count_mismatches") == 0,
        "sealed prompt-budget/range declaration drift",
    )
    proof_by_id = {str(row["neutral_sample_id"]): row for row in proofs}
    ledger_by_id = {str(row["sample_id"]): row for row in ledger}
    for neutral_id, proof in proof_by_id.items():
        neutral = ledger_by_id.get(neutral_id)
        full = ledger_by_id.get(str(proof["full_sample_id"]))
        _require(neutral is not None and full is not None, f"neutral proof ledger join failed: {neutral_id}")
        _require(
            neutral["prompt_token_count"] == proof["neutral_prompt_token_count"]
            and full["prompt_token_count"] == proof["full_prompt_token_count"]
            and neutral["prompt_token_ids_sha256"] == proof["neutral_prompt_token_ids_sha256"]
            and full["prompt_token_ids_sha256"] == proof["full_prompt_token_ids_sha256"],
            f"neutral proof/ledger binding drift: {neutral_id}",
        )

    if verify_static_bindings:
        source = load_and_validate_frozen_source()
        bindings = verify_paper_chk2_bindings(load_tokenizer=False)
        source_by_id = source.by_sample_id
        _require(tuple(row["sample_id"] for row in ledger) == tuple(row["sample_id"] for row in source.ordered_rows), "prepared/source order drift")
        for row in ledger:
            original = source_by_id[str(row["sample_id"])]
            for key in (
                "meeting_id",
                "meeting_start_date",
                "meeting_rank",
                "arm",
                "intervention_topic",
                "variant_rank",
                "prompt_sha256",
                "source_analysis_sha256",
                "reference_minutes_sha256",
                "full_source_analysis_sha256",
                "full_reference_sha256",
            ):
                _require(row.get(key) == original.get(key), f"prepared/source {key} drift: {row['sample_id']}")
            _require(row["paired_seed_key"] == source.full_sample_id_by_meeting[str(row["meeting_id"])], f"paired-seed key drift: {row['sample_id']}")
        expected_audit = build_recorded_lineage_holdout_audit(source)
        _require(manifest.get("recorded_lineage_holdout_audit") == expected_audit, "recorded-lineage holdout audit drift")
        paper_manifest_bindings = ((manifest.get("bindings") or {}).get("paper_chk2") or {})
        _require(set(paper_manifest_bindings) == set(bindings.file_bindings), "paper chk2 binding inventory drift")
        for key, record in bindings.file_bindings.items():
            _require(paper_manifest_bindings.get(key) == dict(record), f"paper chk2 binding drift: {key}")
    return PreparedArtifacts(
        manifest=manifest,
        ledger_rows=tuple(ledger),
        neutral_proofs=tuple(proofs),
        preparation_dir=preparation,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        neutral_proofs_path=proof_path,
    )


def load_prepared_manifest(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Load the sealed preparation manifest after local-file validation."""

    return validate_preparation_artifacts(
        output_root, verify_static_bindings=False
    ).manifest


def load_prepared_ledger(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Return ``(sealed manifest, ordered ledger rows)`` after validation."""

    prepared = validate_preparation_artifacts(output_root, verify_static_bindings=False)
    return prepared.manifest, prepared.ledger_rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument(
        "--skip-static-bindings",
        action="store_true",
        help="For validate only: verify sealed local preparation files without rehashing sources/models.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepared = build_preparation_artifacts(args.output_root)
    else:
        prepared = validate_preparation_artifacts(
            args.output_root,
            verify_static_bindings=not args.skip_static_bindings,
        )
    print(
        json.dumps(
            {
                "status": "valid",
                "evaluation_id": prepared.manifest["evaluation_id"],
                "manifest_path": str(prepared.manifest_path),
                "manifest_payload_sha256": prepared.manifest["integrity"]["payload_sha256"],
                "prompts": len(prepared.ledger_rows),
                "neutral_proofs": len(prepared.neutral_proofs),
                "generation_rows": EXPECTED_GENERATIONS,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "EVALUATION_ID",
    "EXPECTED_GENERATIONS",
    "EXPECTED_MAX_PROMPT_TOKENS",
    "EXPECTED_MEETINGS",
    "EXPECTED_MIN_PROMPT_TOKENS",
    "EXPECTED_PROMPTS",
    "MAX_MODEL_LEN",
    "MAX_NEW_TOKENS",
    "MAX_PROMPT_TOKENS",
    "PaperChk2Bindings",
    "PaperChk2Core8PreparationError",
    "PreparedArtifacts",
    "REPETITION_PENALTY",
    "REPLICATE_SEEDS",
    "TEMPERATURE",
    "TOPICS",
    "TOP_K",
    "TOP_P",
    "VARIANTS_PER_MEETING",
    "FrozenSource",
    "build_exact_token_ledger",
    "build_neutral_token_match_proofs",
    "build_preparation_artifacts",
    "build_recorded_lineage_holdout_audit",
    "canonical_case_coordinates",
    "load_and_validate_frozen_source",
    "load_prepared_ledger",
    "load_prepared_manifest",
    "row_seed_for_case",
    "validate_preparation_artifacts",
    "verify_paper_chk2_bindings",
]
