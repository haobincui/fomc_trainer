"""Fail-closed binding for immutable, audited SFT dataset releases."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from open_r1.provenance import fingerprint_artifact_path


CLEAN_SFT_RELEASE_SCHEMA = "chk1-clean-sft-release-v2"
CLEAN_SFT_AUDIT_SCHEMA = "chk1-clean-sft-source-audit-v1"
CLEAN_SFT_VALIDATION_SCHEMA = "chk1-clean-sft-validator-v1"
CLEAN_SFT_DATA_CONTRACT = "chk1-clean-sft-data-contract-v2"
CLEAN_SFT_TOKEN_CONTRACT = "chk1-sft-single-bos-completion-mask-v1"
STANDALONE_CHK3_RELEASE_SCHEMA = "chk3-minutes-training-release-v1"
STANDALONE_CHK3_DATASET_ROLE = "standalone_chk3_minutes_alignment"
STANDALONE_CHK3_DIRECT_SCOPE = "chk1-to-chk3-direct-sft-non-promotable-v1"
STANDALONE_CHK3_BINDING_SCHEMA = "standalone-chk3-direct-sft-binding-v1"
PAPER_CHK2_RELEASE_SCHEMA = "paper-chk2-downstream-recovery-release-v1"
PAPER_CHK2_HANDOFF_SCHEMA = "paper-chk2-downstream-recovery-release-handoff-v1"
PAPER_CHK2_DATASET_ROLE = (
    "paper_chk2_chk1_final_analysis_to_synthetic_minutes_sft_v6_recovery"
)
PAPER_CHK2_TRAINING_SCOPE = "paper-chk2-chk1-cp200-minutes-sft-v6-downstream128"
PAPER_CHK2_BINDING_SCHEMA = "paper-chk2-minutes-sft-runtime-binding-v1"
PAPER_CHK2_PROMPT_CONTRACT_SCHEMA = "paper-chk2-student-prompt-contract-v1"
PAPER_CHK2_STUDENT_SYSTEM_PROMPT = """\
You are a Federal Reserve Minutes editor. The user supplies a complete
economic or financial analysis. Use the native reasoning section for the
complete reasoning process, including any useful deliberation about the task,
prompt, JSON transport, answer contract, length, or drafting. Within that full
reasoning trace, identify every substantive claim, quantity, date, direction,
comparison, attribution, causal relation, and expression of uncertainty that
the formal rewrite must preserve. Then express the same information as exactly
one formal FOMC Minutes paragraph.

Do not add, remove, broaden, narrow, or contradict any substantive claim.
Preserve the numeric-quantity multiset and every explicit calendar reference.
The final paragraph may reuse phrases, sentences, or extensive wording from
the analysis when that wording is already suitable; lexical overlap is not an
error. The final paragraph as a whole must not be a verbatim copy of the whole
analysis. Do not emit headings, lists, JSON, citations, answer tags, or
model-control tags in the final paragraph.
"""
PAPER_CHK2_STUDENT_SYSTEM_PROMPT_SHA256 = (
    "4730a4ed585238547447ab850836db5a9fc67e5c3b1328b485c88d701ae78c4e"
)
PAPER_CHK2_USER_PROMPT_TEMPLATE = (
    "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
    '{"analysis":"[SOURCE_ANALYSIS]"}'
)
PAPER_CHK2_USER_PROMPT_TEMPLATE_SHA256 = (
    "423e79849cb66d6361c986f705ea4d4b16a03e28fec5826113eb9c8030a976d0"
)
PAPER_CHK2_USER_PROMPT_PREFIX = (
    "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
)
PAPER_CHK2_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'’][A-Za-z0-9]+)*")
PAPER_CHK2_PARENT_MODEL_RELATIVE = (
    "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
)
PAPER_CHK2_PARENT_MODEL_SHA256 = (
    "0989b94792f8ab6377e2979aaedeb37010b9a2c5e05c77a459b4e5cda5b806e3"
)
PAPER_CHK2_PARENT_MANIFEST_SCHEMA = "chk1-cp200-merged-checkpoint-manifest-v1"
PAPER_CHK2_PARENT_MANIFEST_FILE_SHA256 = (
    "59758ab4d2b54592f632776c14d2b06c21c56416e4eb83e09d896f6a06d0a687"
)
PAPER_CHK2_PARENT_MANIFEST_RELATIVE = (
    "docs/summary/20260810T124500Z/chk1_cp200_merge_for_chk2/"
    "checkpoint_manifest.json"
)
PAPER_CHK2_PARENT_AUTHORIZATION_SCHEMA = "chk1-cp200-to-chk2-override-authorization-v1"
PAPER_CHK2_PARENT_AUTHORIZATION_SHA256 = (
    "c8b400ec48be0fd31929f07dea3d30422a2c2638306ab55f4e2e264cb161cb3c"
)
PAPER_CHK2_PARENT_AUTHORIZATION_FILE_SHA256 = (
    "e4e80fbfdd805b3691b3b05fbf3e3f26353eeef399fb7a08f156260e7f0ca554"
)
PAPER_CHK2_PARENT_AUTHORIZATION_RELATIVE = (
    "docs/summary/20260810T124500Z/chk1_cp200_merge_for_chk2/"
    "chk1_cp200_to_chk2_authorization.json"
)
CHK4_DECISION_RELEASE_SCHEMA = "chk4-decision-training-release-v1"
CHK4_DECISION_HANDOFF_SCHEMA = "chk4-decision-training-handoff-v1"
CHK4_DECISION_INPUT_CONTRACT_SCHEMA = "chk4-target-decision-blind-input-contract-v1"
CHK4_DECISION_ROLES = frozenset({"decision_sft", "decision_grpo"})
CHK4_HIER_BALANCED_RELEASE_SCHEMA = "chk4-decision-sft-hier-balanced-release-v1"
CHK4_HIER_BALANCED_HANDOFF_SCHEMA = "chk4-decision-sft-hier-balanced-handoff-v1"
CHK4_HIER_BALANCED_SCHEDULE_SCHEMA = "chk4-decision-sft-fixed-schedule-row-v1"
CHK4_HIER_BALANCED_DATASET_ROLE = "decision_sft_hier_balanced"
CHK4_HIER_BALANCED_SAMPLER_TYPE = "manifest_fixed_schedule_v1"
CHK4_HIER_BALANCED_RELEASE_ID = "chk4_decision_sft_hier_balanced_v1_20260811"
CHK4_HIER_BALANCED_PARENT_RELEASE_ID = "chk4_decision_warmstart_grpo_core_v3_20260810"
CHK4_HIER_BALANCED_PARENT_MANIFEST_SHA256 = (
    "8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893"
)
CHK4_HIER_BALANCED_PARENT_UNIQUE_SHA256 = (
    "1e7ed2c451a29b3807a780c715d46c274c9382eeaf98b1e63da65e121048b0b7"
)
CHK4_HIER_BALANCED_PARENT_SPLIT_SHA256 = {
    "train": "0b7ffb7a3c1f54f8c2dc65f2badbf213ea9fdd0d74e462a7a060ffa5693d56b6",
    "validation": "8913a2c31087ddbf1f678e7dbed975c2ff466254fcf13dd214ed322b5323045e",
    "test": "f89475752af03d58f183ee5182159431cd8da8f66d4efb909c346e41c4103141",
}
CHK4_HIER_BALANCED_PARENT_SPLIT_ROWS = {
    "train": 141,
    "validation": 13,
    "test": 13,
}
CHK4_HIER_BALANCED_DIRECTION_COUNTS = {"hold": 96, "hike": 48, "cut": 48}
CHK4_HIER_BALANCED_WINDOW_COUNTS = {"hold": 4, "hike": 2, "cut": 2}
CHK4_HIER_BALANCED_ROWS = 192
CHK4_HIER_BALANCED_EFFECTIVE_BATCH = 8
CHK4_HIER_BALANCED_STEPS = 24
CHK4_PRE2009_AUGMENTED_RELEASE_SCHEMA = (
    "chk4-decision-pre2009-train-balanced-release-v1"
)
CHK4_PRE2009_AUGMENTED_RELEASE_ID = "chk4_decision_pre2009_train_balanced_v1_20260811"
CHK4_PRE2009_SFT_ROLE = "decision_sft_pre2009_balanced"
CHK4_PRE2009_GRPO_ROLE = "decision_grpo_pre2009_balanced"
CHK4_PRE2009_ROLES = frozenset({CHK4_PRE2009_SFT_ROLE, CHK4_PRE2009_GRPO_ROLE})
CHK4_PRE2009_PHYSICAL_ROLE = {
    CHK4_PRE2009_SFT_ROLE: "decision_sft",
    CHK4_PRE2009_GRPO_ROLE: "decision_grpo",
}
CHK4_PRE2009_SAMPLER_TYPE = "manifest_fixed_schedule_v2"
CHK4_PRE2009_CORRECTION_RELEASE_SCHEMA = (
    "chk4-decision-pre2009-correction-sft-release-v1"
)
CHK4_PRE2009_CORRECTION_RELEASE_ID = (
    "chk4_decision_pre2009_correction_sft_v1_20260811"
)
CHK4_PRE2009_CORRECTION_SFT_ROLE = "decision_sft_pre2009_correction"
CHK4_PRE2009_CORRECTION_SAMPLER_TYPE = "manifest_fixed_schedule_v3"
CHK4_STUDENT_SYSTEM_PROMPT = """\
You are an FOMC policy decision analyst. Use only the supplied target-neutral
pre-meeting analysis. In the native reasoning section, weigh the evidence under
the maximum-employment and price-stability objectives.

Do not use outside or remembered historical information, infer the meeting
identity, or claim that the Committee actually took an action. After closing
the reasoning section, output exactly one JSON object with keys direction and
magnitude_bp. Direction must be cut, hold, or hike. Hold requires magnitude 0;
cut and hike require 25, 50, 75, or 100. Output no headings, commentary, answer
tags, or additional keys.
"""
CHK4_STUDENT_SYSTEM_PROMPT_SHA256 = (
    "426b64532dfb955a3941db2cc2bcfc9a785c0540ffba9ee94afabc189fe4ce18"
)
CHK1_OVERRIDE_CANDIDATE_SCHEMA = "chk1-clean-sft-candidate-v2"
CHK1_OVERRIDE_AUTHORIZATION_SCHEMA = "chk1-sft-semantic-override-authorization-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SPLITS = ("train", "eval", "test")
_CHK3_SPLITS = ("train", "validation", "test")
_CHK1_OVERRIDE_SPLIT_COUNTS = {"train": 1354, "eval": 199, "test": 190}
_CHK1_OVERRIDE_CHANGED_ROWS = 237
_CHK1_OVERRIDE_ACKNOWLEDGEMENTS = {
    "semantic_audit_failed",
    "semantic_audit_contains_blocking_violations",
    "semantic_audit_contains_judge_errors",
    "not_authorized_for_downstream_training",
}
_BLOCKING_KINDS = {"factual", "numerical", "causal", "target_leakage"}
_JUDGE_MODEL = "Qwen3.5-9B"
_RUBRIC_KEYS = {
    "data_fidelity",
    "trend_reasoning",
    "policy_relevance",
    "uncertainty_calibration",
    "fomc_style",
}


class DatasetReleaseValidationError(RuntimeError):
    """Raised when a configured immutable dataset release cannot be trusted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_path(path: str | Path, *, label: str, directory: bool) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).absolute()
    if candidate.is_symlink():
        raise DatasetReleaseValidationError(f"{label} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise DatasetReleaseValidationError(f"{label} is missing: {candidate}") from exc
    if candidate != resolved:
        raise DatasetReleaseValidationError(
            f"{label} must be a canonical path without symlink components"
        )
    if directory and not resolved.is_dir():
        raise DatasetReleaseValidationError(f"{label} must be a directory")
    if not directory and not resolved.is_file():
        raise DatasetReleaseValidationError(f"{label} must be a file")
    return resolved


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetReleaseValidationError(f"{label} must be an object")
    return value


def _required_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DatasetReleaseValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _required_count(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetReleaseValidationError(f"{label} must be a non-negative integer")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl_objects(path: Path, *, label: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                line = raw_line.rstrip("\n")
                if not line or line.endswith("\r"):
                    raise DatasetReleaseValidationError(
                        f"{label}:{line_number} has invalid JSONL framing"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetReleaseValidationError(
                        f"{label}:{line_number} is invalid JSON"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise DatasetReleaseValidationError(
                        f"{label}:{line_number} must be an object"
                    )
                rows.append(value)
    except (OSError, UnicodeError) as exc:
        raise DatasetReleaseValidationError(f"cannot read {label}") from exc
    return rows


def _read_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetReleaseValidationError(f"{label} is invalid JSON") from exc
    return _required_mapping(value, label=label)


def _resolve_release_member(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise DatasetReleaseValidationError(
            f"{label} must be a non-empty release-relative path"
        )
    candidate = root / value
    resolved = _canonical_path(candidate, label=label, directory=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DatasetReleaseValidationError(
            f"{label} escapes the release root"
        ) from exc
    return resolved


def _validate_chk4_decision(
    direction: Any, magnitude_bp: Any, *, label: str
) -> tuple[str, int]:
    if direction not in {"cut", "hold", "hike"}:
        raise DatasetReleaseValidationError(f"{label} has an invalid direction")
    if isinstance(magnitude_bp, bool) or not isinstance(magnitude_bp, int):
        raise DatasetReleaseValidationError(f"{label} has an invalid magnitude_bp")
    allowed = {0} if direction == "hold" else {25, 50, 75, 100}
    if magnitude_bp not in allowed:
        raise DatasetReleaseValidationError(f"{label} has an invalid decision")
    return str(direction), magnitude_bp


def _verify_chk4_split_schemas(
    *, release_root: Path, physical_counts: Mapping[str, Any]
) -> None:
    """Validate both role schemas and their row-wise prompt/label parity."""

    for split in ("train", "validation", "test"):
        expected_rows = _required_count(
            physical_counts.get(split), label=f"physical_split_counts.{split}"
        )
        if expected_rows <= 0:
            raise DatasetReleaseValidationError(
                f"chk4 {split} split must contain at least one row"
            )
        sft_rows = _read_jsonl_objects(
            release_root / "decision_sft" / f"{split}.jsonl",
            label=f"decision_sft/{split}.jsonl",
        )
        grpo_rows = _read_jsonl_objects(
            release_root / "decision_grpo" / f"{split}.jsonl",
            label=f"decision_grpo/{split}.jsonl",
        )
        if len(sft_rows) != expected_rows or len(grpo_rows) != expected_rows:
            raise DatasetReleaseValidationError(
                f"chk4 {split} role row counts disagree with the manifest"
            )
        for row_number, (sft, grpo) in enumerate(
            zip(sft_rows, grpo_rows, strict=True), start=1
        ):
            label = f"chk4 {split} row {row_number}"
            if set(sft) != {"prompt", "response"}:
                raise DatasetReleaseValidationError(
                    f"{label} has an invalid Decision-SFT schema"
                )
            if set(grpo) != {
                "sample_id",
                "prompt",
                "direction",
                "magnitude_bp",
            }:
                raise DatasetReleaseValidationError(
                    f"{label} has an invalid Decision-GRPO schema"
                )
            prompt = sft.get("prompt")
            response = sft.get("response")
            sample_id = grpo.get("sample_id")
            if (
                not isinstance(prompt, str)
                or not prompt.strip()
                or not isinstance(response, str)
                or not response.strip()
                or not isinstance(sample_id, str)
                or not sample_id.strip()
            ):
                raise DatasetReleaseValidationError(
                    f"{label} contains an empty required field"
                )
            if grpo.get("prompt") != prompt:
                raise DatasetReleaseValidationError(
                    f"{label} has SFT/GRPO prompt drift"
                )
            if response.count("\n</think>\n") != 1 or "<think>" in response:
                raise DatasetReleaseValidationError(
                    f"{label} has an invalid native reasoning boundary"
                )
            reasoning, answer = response.split("\n</think>\n", 1)
            if not reasoning.strip() or not answer.strip():
                raise DatasetReleaseValidationError(
                    f"{label} has an empty reasoning or final answer"
                )
            try:
                decision = json.loads(answer)
            except json.JSONDecodeError as exc:
                raise DatasetReleaseValidationError(
                    f"{label} final answer is not JSON"
                ) from exc
            if not isinstance(decision, Mapping) or set(decision) != {
                "direction",
                "magnitude_bp",
            }:
                raise DatasetReleaseValidationError(
                    f"{label} final answer has an invalid decision schema"
                )
            sft_decision = _validate_chk4_decision(
                decision.get("direction"),
                decision.get("magnitude_bp"),
                label=f"{label} SFT target",
            )
            grpo_decision = _validate_chk4_decision(
                grpo.get("direction"),
                grpo.get("magnitude_bp"),
                label=f"{label} GRPO target",
            )
            if sft_decision != grpo_decision:
                raise DatasetReleaseValidationError(
                    f"{label} has SFT/GRPO target drift"
                )


def _verify_chk4_model_tokenizer(
    *, release_root: Path, release: Mapping[str, Any], model_path: str | Path
) -> dict[str, Any]:
    """Bind the current parent model to the release-audited tokenizer bytes."""

    model = _canonical_path(model_path, label="chk4 model_name_or_path", directory=True)
    sources = _required_mapping(release.get("sources"), label="chk4 release sources")
    tokenizer = _required_mapping(
        sources.get("tokenizer"), label="chk4 release tokenizer source"
    )
    tokenizer_files = _required_mapping(
        tokenizer.get("files"), label="chk4 release tokenizer files"
    )
    if not tokenizer_files:
        raise DatasetReleaseValidationError("chk4 release tokenizer file set is empty")
    normalized_files: dict[str, dict[str, Any]] = {}
    for relative_path, raw_descriptor in tokenizer_files.items():
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
        ):
            raise DatasetReleaseValidationError(
                "chk4 release tokenizer contains an invalid path"
            )
        descriptor = _required_mapping(
            raw_descriptor, label=f"chk4 tokenizer files.{relative_path}"
        )
        if set(descriptor) != {"bytes", "sha256"}:
            raise DatasetReleaseValidationError(
                f"chk4 tokenizer descriptor schema drift: {relative_path}"
            )
        candidate = model / relative_path
        candidate = _canonical_path(
            candidate,
            label=f"chk4 model tokenizer file {relative_path}",
            directory=False,
        )
        try:
            candidate.relative_to(model)
        except ValueError as exc:
            raise DatasetReleaseValidationError(
                f"chk4 model tokenizer path escapes the model: {relative_path}"
            ) from exc
        expected_bytes = _required_count(
            descriptor.get("bytes"),
            label=f"chk4 tokenizer files.{relative_path}.bytes",
        )
        expected_sha = _required_sha256(
            descriptor.get("sha256"),
            label=f"chk4 tokenizer files.{relative_path}.sha256",
        )
        if candidate.stat().st_size != expected_bytes:
            raise DatasetReleaseValidationError(
                f"chk4 model tokenizer byte-size drift: {relative_path}"
            )
        if sha256_file(candidate) != expected_sha:
            raise DatasetReleaseValidationError(
                f"chk4 model tokenizer hash drift: {relative_path}"
            )
        normalized_files[relative_path] = {
            "bytes": expected_bytes,
            "sha256": expected_sha,
        }

    data_quality = _read_json_object(
        release_root / "audits/data_quality.json", label="chk4 data-quality audit"
    )
    token_contract = _required_mapping(
        data_quality.get("token_contract"), label="chk4 token contract"
    )
    if token_contract.get("tokenizer_files") != normalized_files:
        raise DatasetReleaseValidationError(
            "chk4 release tokenizer disagrees with the audited token contract"
        )
    return {
        "model_path": str(model),
        "files": normalized_files,
        "bundle_sha256": _sha256_text(_canonical_json(normalized_files)),
    }


def verify_chk4_decision_release(
    *,
    dataset_dir: str | Path,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    dataset_role: str,
    system_prompt: str | None,
    model_path: str | Path,
) -> dict[str, Any]:
    """Lightweight runtime binding for the sealed chk4 Decision release.

    The publisher performs the expensive source-lineage replay once. At train
    time this verifier pins the externally supplied manifest hash, verifies
    every sealed release member, and exposes only train/validation for the
    configured role. The sealed test split is checked but never returned to
    the training loader.
    """

    if dataset_role not in CHK4_DECISION_ROLES:
        raise DatasetReleaseValidationError(
            "dataset_chk4_role must be decision_sft or decision_grpo"
        )
    expected_sha = _required_sha256(
        expected_manifest_sha256,
        label="dataset_chk4_release_manifest_sha256",
    )
    if system_prompt != CHK4_STUDENT_SYSTEM_PROMPT:
        raise DatasetReleaseValidationError(
            "chk4 system_prompt does not match the sealed Decision release"
        )
    if _sha256_text(CHK4_STUDENT_SYSTEM_PROMPT) != (CHK4_STUDENT_SYSTEM_PROMPT_SHA256):
        raise DatasetReleaseValidationError("embedded chk4 system prompt digest drift")

    dataset = _canonical_path(dataset_dir, label="dataset_name", directory=True)
    manifest = _canonical_path(
        manifest_path, label="dataset_chk4_release_manifest", directory=False
    )
    release_root = manifest.parent
    if dataset != release_root / dataset_role:
        raise DatasetReleaseValidationError(
            "dataset_name must be the configured chk4 role directory beside the manifest"
        )
    if sha256_file(manifest) != expected_sha:
        raise DatasetReleaseValidationError(
            "chk4 release manifest SHA-256 disagrees with the training config"
        )

    root = _read_json_object(manifest, label="chk4 release manifest")
    if root.get("schema_version") != CHK4_DECISION_RELEASE_SCHEMA:
        raise DatasetReleaseValidationError("chk4 Decision release schema drift")
    if (
        root.get("quality_status") != "passed"
        or root.get("immutable") is not True
        or root.get("training_ready") is not True
    ):
        raise DatasetReleaseValidationError(
            "chk4 Decision release must be immutable, passed, and training-ready"
        )
    if root.get("canonical_dag_bindable") is not False:
        raise DatasetReleaseValidationError(
            "chk4 direct branch release must remain outside the canonical DAG"
        )
    expected_semantic_assurance = {
        "structural_lineage_replay": "passed",
        "known_pattern_target_outcome_hits": 0,
        "independent_source_only_semantic_judge": "not_run",
    }
    if root.get("semantic_assurance") != expected_semantic_assurance:
        raise DatasetReleaseValidationError(
            "chk4 release is not the structurally replayed v3 contract"
        )
    sources = _required_mapping(root.get("sources"), label="chk4 release sources")
    canonical_source = _required_mapping(
        sources.get("canonical_chk1_analysis_source"),
        label="chk4 canonical chk1 source",
    )
    if (
        canonical_source.get("source_rows") != 2072
        or canonical_source.get("evidence_rows_checked") != 16212
        or _SHA256_RE.fullmatch(str(canonical_source.get("tree_sha256") or "")) is None
    ):
        raise DatasetReleaseValidationError(
            "chk4 v3 canonical chk1 lineage binding drift"
        )
    publisher_source = _required_mapping(
        sources.get("publisher"), label="chk4 publisher source"
    )
    if (
        publisher_source.get("snapshot") != "provenance/publisher_snapshot.py"
        or publisher_source.get("sft_prompt_renderer_snapshot")
        != "provenance/sft_prompt_renderer_snapshot.py"
    ):
        raise DatasetReleaseValidationError("chk4 v3 publisher snapshot binding drift")
    release_id = root.get("release_id")
    if (
        not isinstance(release_id, str)
        or not release_id
        or release_id != release_root.name
    ):
        raise DatasetReleaseValidationError(
            "chk4 release identity does not match its directory"
        )
    if root.get("test_is_sealed_evaluation_only") is not True:
        raise DatasetReleaseValidationError(
            "chk4 test split is not sealed evaluation-only"
        )

    files = _required_mapping(root.get("files"), label="chk4 release files")
    actual_files: set[str] = set()
    for path in release_root.rglob("*"):
        if path.is_symlink():
            raise DatasetReleaseValidationError(
                f"chk4 release contains a symlink: {path}"
            )
        mode = path.stat().st_mode & 0o777
        expected_mode = 0o444 if path.is_file() else 0o555
        if mode != expected_mode:
            raise DatasetReleaseValidationError(
                f"chk4 release member has mutable mode: {path}"
            )
        if path.is_file():
            actual_files.add(path.relative_to(release_root).as_posix())
    if (release_root.stat().st_mode & 0o777) != 0o555:
        raise DatasetReleaseValidationError("chk4 release root has mutable mode")
    if actual_files != set(files) | {"release_manifest.json", "handoff.json"}:
        raise DatasetReleaseValidationError(
            "chk4 release contains missing or unlisted files"
        )
    for relative_path, raw_descriptor in files.items():
        if not isinstance(relative_path, str):
            raise DatasetReleaseValidationError("chk4 release file key is invalid")
        descriptor = _required_mapping(
            raw_descriptor, label=f"chk4 release files.{relative_path}"
        )
        expected_descriptor_keys = {"path", "bytes", "sha256"}
        if relative_path.endswith(".jsonl"):
            expected_descriptor_keys.add("rows")
        if (
            set(descriptor) != expected_descriptor_keys
            or descriptor.get("path") != relative_path
        ):
            raise DatasetReleaseValidationError(
                f"chk4 release file descriptor schema drift: {relative_path}"
            )
        path = _resolve_release_member(
            release_root,
            descriptor.get("path"),
            label=f"chk4 release files.{relative_path}.path",
        )
        if path.stat().st_size != _required_count(
            descriptor.get("bytes"), label=f"chk4 release files.{relative_path}.bytes"
        ):
            raise DatasetReleaseValidationError(
                f"chk4 release file byte-size drift: {relative_path}"
            )
        if sha256_file(path) != _required_sha256(
            descriptor.get("sha256"),
            label=f"chk4 release files.{relative_path}.sha256",
        ):
            raise DatasetReleaseValidationError(
                f"chk4 release file hash drift: {relative_path}"
            )
        if relative_path.endswith(".jsonl"):
            expected_rows = _required_count(
                descriptor.get("rows"),
                label=f"chk4 release files.{relative_path}.rows",
            )
            if len(_read_jsonl_objects(path, label=relative_path)) != expected_rows:
                raise DatasetReleaseValidationError(
                    f"chk4 release file row-count drift: {relative_path}"
                )

    handoff_record = _required_mapping(
        root.get("handoff"), label="chk4 handoff binding"
    )
    if (
        handoff_record.get("path") != "handoff.json"
        or handoff_record.get("schema_version") != CHK4_DECISION_HANDOFF_SCHEMA
    ):
        raise DatasetReleaseValidationError("chk4 handoff manifest binding drift")
    handoff = _read_json_object(release_root / "handoff.json", label="chk4 handoff")
    if (
        handoff.get("schema_version") != CHK4_DECISION_HANDOFF_SCHEMA
        or handoff.get("release_id") != release_id
        or handoff.get("quality_status") != "passed"
        or handoff.get("immutable") is not True
        or handoff.get("training_ready") is not True
        or handoff.get("test_is_sealed_evaluation_only") is not True
        or handoff.get("release_manifest") != "release_manifest.json"
        or handoff.get("release_manifest_sha256") != expected_sha
        or handoff.get("decision_sft_path") != "decision_sft"
        or handoff.get("decision_grpo_path") != "decision_grpo"
    ):
        raise DatasetReleaseValidationError("chk4 handoff contract drift")
    unsigned_handoff = dict(handoff)
    unsigned_handoff.pop("release_manifest_sha256", None)
    if handoff_record.get("unsigned_payload_sha256") != _sha256_text(
        _canonical_json(unsigned_handoff)
    ):
        raise DatasetReleaseValidationError("chk4 handoff unsigned payload drift")

    input_record = _required_mapping(
        root.get("input_contract"), label="chk4 input contract binding"
    )
    if (
        input_record.get("path") != "contracts/decision_input_contract.json"
        or input_record.get("schema_version") != CHK4_DECISION_INPUT_CONTRACT_SCHEMA
    ):
        raise DatasetReleaseValidationError("chk4 input contract manifest drift")
    contract = _read_json_object(
        release_root / "contracts/decision_input_contract.json",
        label="chk4 decision input contract",
    )
    unsigned_contract = dict(contract)
    contract_sha = unsigned_contract.pop("contract_sha256", None)
    if (
        contract.get("schema_version") != CHK4_DECISION_INPUT_CONTRACT_SCHEMA
        or contract.get("status") != "active"
        or contract_sha != _sha256_text(_canonical_json(unsigned_contract))
        or input_record.get("contract_sha256") != contract_sha
        or contract.get("student_system_prompt") != CHK4_STUDENT_SYSTEM_PROMPT
        or contract.get("student_system_prompt_sha256")
        != CHK4_STUDENT_SYSTEM_PROMPT_SHA256
    ):
        raise DatasetReleaseValidationError("chk4 system/input contract drift")
    source_summary = _read_json_object(
        release_root / "provenance/source_materialization_summary.json",
        label="chk4 source materialization summary",
    )
    if source_summary.get("system_prompt_sha256") != (
        CHK4_STUDENT_SYSTEM_PROMPT_SHA256
    ):
        raise DatasetReleaseValidationError("chk4 source system prompt digest drift")
    point_in_time = _read_json_object(
        release_root / "audits/point_in_time_lineage.json",
        label="chk4 point-in-time lineage audit",
    )
    point_canonical = _required_mapping(
        point_in_time.get("canonical_chk1"),
        label="chk4 point-in-time canonical chk1 audit",
    )
    if (
        point_in_time.get("status") != "passed"
        or point_canonical.get("source_rows") != 2072
        or point_canonical.get("evidence_rows_checked") != 16212
        or point_canonical.get("future_evidence_violations") != 0
    ):
        raise DatasetReleaseValidationError("chk4 v3 point-in-time audit drift")

    roles = _required_mapping(root.get("training_roles"), label="chk4 training_roles")
    if set(roles) != CHK4_DECISION_ROLES:
        raise DatasetReleaseValidationError("chk4 training role set drift")
    expected_role_contracts = {
        "decision_sft": {
            "dataset_path": "decision_sft",
            "parent_role": "selected_chk1_merged",
            "max_length": 3072,
            "completion_only_loss": True,
        },
        "decision_grpo": {
            "dataset_path": "decision_grpo",
            "parent_role": "merged_decision_sft_warm_start",
            "reward": "decision_dense_v2",
            "max_prompt_length": 2560,
            "max_completion_length": 512,
        },
    }
    if {name: dict(value) for name, value in roles.items()} != expected_role_contracts:
        raise DatasetReleaseValidationError("chk4 training role contract drift")
    physical = _required_mapping(
        root.get("physical_split_counts"), label="chk4 physical_split_counts"
    )
    if set(physical) != CHK4_DECISION_ROLES:
        raise DatasetReleaseValidationError("chk4 physical role counts drift")
    sft_counts = _required_mapping(
        physical.get("decision_sft"), label="chk4 Decision-SFT counts"
    )
    grpo_counts = _required_mapping(
        physical.get("decision_grpo"), label="chk4 Decision-GRPO counts"
    )
    if set(sft_counts) != {"train", "validation", "test"} or dict(sft_counts) != dict(
        grpo_counts
    ):
        raise DatasetReleaseValidationError("chk4 role split counts drift")
    _verify_chk4_split_schemas(release_root=release_root, physical_counts=sft_counts)
    tokenizer_binding = _verify_chk4_model_tokenizer(
        release_root=release_root,
        release=root,
        model_path=model_path,
    )

    return {
        "schema_version": "chk4-decision-runtime-binding-v1",
        "release_id": release_id,
        "dataset_role": dataset_role,
        "release_manifest_sha256": expected_sha,
        "split_files": {
            "train": release_root / dataset_role / "train.jsonl",
            "validation": release_root / dataset_role / "validation.jsonl",
        },
        "test_verified_but_not_loaded": True,
        "system_prompt_sha256": CHK4_STUDENT_SYSTEM_PROMPT_SHA256,
        "tokenizer_binding": tokenizer_binding,
    }


def _resolve_repository_member(value: Any, *, label: str) -> Path:
    """Resolve a repository-relative lineage path without trusting the CWD."""

    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise DatasetReleaseValidationError(
            f"{label} must be a non-empty repository-relative path"
        )
    repository_root = Path(__file__).resolve().parents[3]
    resolved = _canonical_path(
        repository_root / value,
        label=label,
        directory=False,
    )
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise DatasetReleaseValidationError(f"{label} escapes the repository") from exc
    return resolved


def _chk4_hier_training_row_id(
    *, source_sample_id: str, source_repeat_index: int, schedule_index: int
) -> str:
    payload = (
        f"{CHK4_HIER_BALANCED_RELEASE_ID}\0"
        f"{CHK4_HIER_BALANCED_SCHEDULE_SCHEMA}\0{source_sample_id}\0"
        f"{source_repeat_index}\0{schedule_index}"
    )
    return "row-" + _sha256_text(payload)[:24]


def verify_chk4_hier_balanced_release(
    *,
    dataset_dir: str | Path,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    dataset_role: str,
    system_prompt: str | None,
    model_path: str | Path,
) -> dict[str, Any]:
    """Verify the immutable fixed-schedule Decision-SFT child release.

    This verifier is intentionally independent of the release materializer. It
    binds the child to the already sealed Decision-v3 parent, validates every
    child byte, replays the 192-row source lineage, and returns absolute files
    plus the exact sampler contract consumed by the trainer.
    """

    if dataset_role != CHK4_HIER_BALANCED_DATASET_ROLE:
        raise DatasetReleaseValidationError(
            "hier-balanced release requires dataset_chk4_role="
            f"{CHK4_HIER_BALANCED_DATASET_ROLE}"
        )
    expected_sha = _required_sha256(
        expected_manifest_sha256,
        label="dataset_chk4_release_manifest_sha256",
    )
    if system_prompt != CHK4_STUDENT_SYSTEM_PROMPT:
        raise DatasetReleaseValidationError(
            "chk4 system_prompt does not match the sealed Decision release"
        )

    dataset = _canonical_path(dataset_dir, label="dataset_name", directory=True)
    manifest = _canonical_path(
        manifest_path,
        label="dataset_chk4_release_manifest",
        directory=False,
    )
    release_root = manifest.parent
    if dataset != release_root / "decision_sft":
        raise DatasetReleaseValidationError(
            "hier-balanced dataset_name must be its decision_sft directory"
        )
    if release_root.name != CHK4_HIER_BALANCED_RELEASE_ID:
        raise DatasetReleaseValidationError("hier-balanced release directory drift")
    if sha256_file(manifest) != expected_sha:
        raise DatasetReleaseValidationError(
            "hier-balanced manifest SHA-256 disagrees with the training config"
        )

    root = _read_json_object(manifest, label="hier-balanced release manifest")
    if (
        root.get("schema_version") != CHK4_HIER_BALANCED_RELEASE_SCHEMA
        or root.get("release_id") != CHK4_HIER_BALANCED_RELEASE_ID
        or root.get("dataset_role") != CHK4_HIER_BALANCED_DATASET_ROLE
        or root.get("quality_status") != "passed"
        or root.get("immutable") is not True
        or root.get("training_ready") is not True
        or root.get("canonical_dag_bindable") is not False
        or root.get("test_is_sealed_evaluation_only") is not True
    ):
        raise DatasetReleaseValidationError(
            "hier-balanced release identity/readiness contract drift"
        )

    files = _required_mapping(root.get("files"), label="hier-balanced files")
    actual_files: set[str] = set()
    for path in release_root.rglob("*"):
        if path.is_symlink():
            raise DatasetReleaseValidationError(
                f"hier-balanced release contains a symlink: {path}"
            )
        expected_mode = 0o444 if path.is_file() else 0o555
        if (path.stat().st_mode & 0o777) != expected_mode:
            raise DatasetReleaseValidationError(
                f"hier-balanced release member has mutable mode: {path}"
            )
        if path.is_file():
            actual_files.add(path.relative_to(release_root).as_posix())
    if (release_root.stat().st_mode & 0o777) != 0o555:
        raise DatasetReleaseValidationError("hier-balanced release root is mutable")
    if actual_files != set(files) | {"release_manifest.json", "handoff.json"}:
        raise DatasetReleaseValidationError(
            "hier-balanced release contains missing or unlisted files"
        )

    normalized_files: dict[str, dict[str, Any]] = {}
    for relative_path, raw_descriptor in files.items():
        if not isinstance(relative_path, str):
            raise DatasetReleaseValidationError(
                "hier-balanced release has a non-string file key"
            )
        descriptor = _required_mapping(
            raw_descriptor,
            label=f"hier-balanced files.{relative_path}",
        )
        expected_keys = {"path", "bytes", "sha256"}
        if relative_path.endswith(".jsonl"):
            expected_keys.add("rows")
        if set(descriptor) != expected_keys or descriptor.get("path") != relative_path:
            raise DatasetReleaseValidationError(
                f"hier-balanced descriptor schema drift: {relative_path}"
            )
        path = _resolve_release_member(
            release_root,
            relative_path,
            label=f"hier-balanced files.{relative_path}.path",
        )
        expected_bytes = _required_count(
            descriptor.get("bytes"),
            label=f"hier-balanced files.{relative_path}.bytes",
        )
        expected_file_sha = _required_sha256(
            descriptor.get("sha256"),
            label=f"hier-balanced files.{relative_path}.sha256",
        )
        if (
            path.stat().st_size != expected_bytes
            or sha256_file(path) != expected_file_sha
        ):
            raise DatasetReleaseValidationError(
                f"hier-balanced file bytes/hash drift: {relative_path}"
            )
        normalized = {
            "path": relative_path,
            "bytes": expected_bytes,
            "sha256": expected_file_sha,
        }
        if relative_path.endswith(".jsonl"):
            expected_rows = _required_count(
                descriptor.get("rows"),
                label=f"hier-balanced files.{relative_path}.rows",
            )
            if len(_read_jsonl_objects(path, label=relative_path)) != expected_rows:
                raise DatasetReleaseValidationError(
                    f"hier-balanced row-count drift: {relative_path}"
                )
            normalized["rows"] = expected_rows
        normalized_files[relative_path] = normalized

    split_files = _required_mapping(
        root.get("split_files"), label="hier-balanced split_files"
    )
    expected_split_files = {
        split: normalized_files[f"decision_sft/{split}.jsonl"]
        for split in ("train", "validation", "test")
    }
    if {
        name: dict(value) for name, value in split_files.items()
    } != expected_split_files:
        raise DatasetReleaseValidationError("hier-balanced split descriptor drift")
    if root.get("unique_split_counts") != {
        "train": 102,
        "validation": 13,
        "test": 13,
    } or root.get("physical_split_counts") != {
        "train": CHK4_HIER_BALANCED_ROWS,
        "validation": 13,
        "test": 13,
    }:
        raise DatasetReleaseValidationError("hier-balanced split counts drift")
    if root.get("training_roles") != {
        CHK4_HIER_BALANCED_DATASET_ROLE: {
            "dataset_path": "decision_sft",
            "parent_role": "selected_chk1_merged",
            "max_length": 3072,
            "completion_only_loss": True,
            "sampler_contract": "sampler_contract",
        }
    }:
        raise DatasetReleaseValidationError("hier-balanced training role drift")

    parent_record = _required_mapping(
        root.get("parent_release"), label="hier-balanced parent_release"
    )
    expected_parent_keys = {
        "release_id",
        "manifest_path",
        "manifest_sha256",
        "unique_train_path",
        "unique_train_sha256",
        "unique_train_rows",
        "decision_sft_split_sha256",
        "decision_sft_split_rows",
    }
    if set(parent_record) != expected_parent_keys:
        raise DatasetReleaseValidationError("hier-balanced parent schema drift")
    if (
        parent_record.get("release_id") != CHK4_HIER_BALANCED_PARENT_RELEASE_ID
        or parent_record.get("manifest_sha256")
        != CHK4_HIER_BALANCED_PARENT_MANIFEST_SHA256
        or parent_record.get("unique_train_sha256")
        != CHK4_HIER_BALANCED_PARENT_UNIQUE_SHA256
        or parent_record.get("unique_train_rows") != 102
        or parent_record.get("decision_sft_split_sha256")
        != CHK4_HIER_BALANCED_PARENT_SPLIT_SHA256
        or parent_record.get("decision_sft_split_rows")
        != CHK4_HIER_BALANCED_PARENT_SPLIT_ROWS
    ):
        raise DatasetReleaseValidationError("hier-balanced parent binding drift")
    parent_manifest = _resolve_repository_member(
        parent_record.get("manifest_path"), label="hier-balanced parent manifest"
    )
    parent_unique = _resolve_repository_member(
        parent_record.get("unique_train_path"),
        label="hier-balanced parent unique train",
    )
    parent_root = parent_manifest.parent
    if (
        parent_root.name != CHK4_HIER_BALANCED_PARENT_RELEASE_ID
        or sha256_file(parent_manifest) != CHK4_HIER_BALANCED_PARENT_MANIFEST_SHA256
        or sha256_file(parent_unique) != CHK4_HIER_BALANCED_PARENT_UNIQUE_SHA256
        or parent_unique != parent_root / "manifests/unique/train.jsonl"
    ):
        raise DatasetReleaseValidationError("hier-balanced parent path/hash drift")
    parent_verified = verify_chk4_decision_release(
        dataset_dir=parent_root / "decision_sft",
        manifest_path=parent_manifest,
        expected_manifest_sha256=CHK4_HIER_BALANCED_PARENT_MANIFEST_SHA256,
        dataset_role="decision_sft",
        system_prompt=system_prompt,
        model_path=model_path,
    )

    copied_unique = release_root / "manifests/parent_unique_train.jsonl"
    if copied_unique.read_bytes() != parent_unique.read_bytes():
        raise DatasetReleaseValidationError(
            "hier-balanced copied unique lineage is not byte-identical to parent"
        )
    for split in ("validation", "test"):
        derived = release_root / "decision_sft" / f"{split}.jsonl"
        parent = parent_root / "decision_sft" / f"{split}.jsonl"
        if derived.read_bytes() != parent.read_bytes():
            raise DatasetReleaseValidationError(
                f"hier-balanced {split} is not byte-inherited from parent"
            )
    input_contract = _required_mapping(
        root.get("input_contract"), label="hier-balanced input_contract"
    )
    if (
        input_contract
        != {
            "path": "contracts/decision_input_contract.json",
            "sha256": normalized_files["contracts/decision_input_contract.json"][
                "sha256"
            ],
        }
        or (release_root / "contracts/decision_input_contract.json").read_bytes()
        != (parent_root / "contracts/decision_input_contract.json").read_bytes()
    ):
        raise DatasetReleaseValidationError("hier-balanced input contract drift")

    unique_rows = _read_jsonl_objects(parent_unique, label="parent unique train")
    if len(unique_rows) != 102:
        raise DatasetReleaseValidationError("hier-balanced parent unique row drift")
    unique_by_id: dict[str, Mapping[str, Any]] = {}
    for number, row in enumerate(unique_rows, 1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in unique_by_id:
            raise DatasetReleaseValidationError(
                f"parent unique train row {number} has invalid sample_id"
            )
        _validate_chk4_decision(
            row.get("direction"),
            row.get("magnitude_bp"),
            label=f"parent unique train row {number}",
        )
        for key in ("prompt_sha256", "response_sha256", "gold_sha256"):
            _required_sha256(row.get(key), label=f"parent unique row {number}.{key}")
        unique_by_id[sample_id] = row

    schedule_path = release_root / "manifests/sampler_schedule.jsonl"
    train_path = release_root / "decision_sft/train.jsonl"
    schedule_rows = _read_jsonl_objects(schedule_path, label="fixed sampler schedule")
    train_rows = _read_jsonl_objects(train_path, label="hier-balanced train")
    if len(schedule_rows) != CHK4_HIER_BALANCED_ROWS or len(train_rows) != (
        CHK4_HIER_BALANCED_ROWS
    ):
        raise DatasetReleaseValidationError("hier-balanced train/schedule size drift")
    schedule_keys = {
        "schema_version",
        "schedule_index",
        "optimizer_step",
        "microbatch_slot",
        "training_row_id",
        "source_sample_id",
        "source_repeat_index",
        "source_repeat_total",
        "direction",
        "magnitude_bp",
        "prompt_sha256",
        "response_sha256",
        "gold_sha256",
    }
    train_keys = {
        "prompt",
        "response",
        "training_row_id",
        "source_sample_id",
        "schedule_index",
    }
    direction_counts = {"hold": 0, "hike": 0, "cut": 0}
    repeat_indexes: dict[str, list[int]] = {}
    repeat_totals: dict[str, int] = {}
    training_ids: set[str] = set()
    for index, (schedule, train) in enumerate(
        zip(schedule_rows, train_rows, strict=True)
    ):
        label = f"hier-balanced schedule row {index}"
        if set(schedule) != schedule_keys or set(train) != train_keys:
            raise DatasetReleaseValidationError(f"{label} schema drift")
        if (
            schedule.get("schema_version") != CHK4_HIER_BALANCED_SCHEDULE_SCHEMA
            or schedule.get("schedule_index") != index
            or schedule.get("optimizer_step")
            != index // CHK4_HIER_BALANCED_EFFECTIVE_BATCH
            or schedule.get("microbatch_slot")
            != index % CHK4_HIER_BALANCED_EFFECTIVE_BATCH
            or train.get("schedule_index") != index
        ):
            raise DatasetReleaseValidationError(f"{label} order/index drift")
        sample_id = schedule.get("source_sample_id")
        source = unique_by_id.get(str(sample_id))
        if source is None or train.get("source_sample_id") != sample_id:
            raise DatasetReleaseValidationError(f"{label} source lineage drift")
        repeat_index = schedule.get("source_repeat_index")
        repeat_total = schedule.get("source_repeat_total")
        if (
            isinstance(repeat_index, bool)
            or not isinstance(repeat_index, int)
            or repeat_index < 0
            or isinstance(repeat_total, bool)
            or not isinstance(repeat_total, int)
            or repeat_total <= 0
        ):
            raise DatasetReleaseValidationError(f"{label} repeat contract drift")
        training_id = schedule.get("training_row_id")
        if (
            training_id
            != _chk4_hier_training_row_id(
                source_sample_id=str(sample_id),
                source_repeat_index=repeat_index,
                schedule_index=index,
            )
            or training_id in training_ids
            or train.get("training_row_id") != training_id
        ):
            raise DatasetReleaseValidationError(f"{label} training row ID drift")
        training_ids.add(str(training_id))
        for key in (
            "direction",
            "magnitude_bp",
            "prompt_sha256",
            "response_sha256",
            "gold_sha256",
        ):
            if schedule.get(key) != source.get(key):
                raise DatasetReleaseValidationError(f"{label} parent {key} drift")
        prompt = train.get("prompt")
        response = train.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise DatasetReleaseValidationError(f"{label} payload type drift")
        if _sha256_text(prompt) != schedule.get("prompt_sha256") or _sha256_text(
            response
        ) != schedule.get("response_sha256"):
            raise DatasetReleaseValidationError(f"{label} payload hash drift")
        if response.count("\n</think>\n") != 1 or "<think>" in response:
            raise DatasetReleaseValidationError(f"{label} reasoning boundary drift")
        reasoning, answer = response.split("\n</think>\n", 1)
        if not reasoning.strip() or answer != answer.strip() or "```" in answer:
            raise DatasetReleaseValidationError(f"{label} answer framing drift")
        try:
            decision = json.loads(answer)
        except json.JSONDecodeError as exc:
            raise DatasetReleaseValidationError(f"{label} invalid answer JSON") from exc
        if not isinstance(decision, Mapping) or set(decision) != {
            "direction",
            "magnitude_bp",
        }:
            raise DatasetReleaseValidationError(f"{label} decision schema drift")
        parsed_decision = _validate_chk4_decision(
            decision.get("direction"),
            decision.get("magnitude_bp"),
            label=label,
        )
        if parsed_decision != (schedule.get("direction"), schedule.get("magnitude_bp")):
            raise DatasetReleaseValidationError(f"{label} response label drift")
        gold = _canonical_json(
            {"direction": parsed_decision[0], "magnitude_bp": parsed_decision[1]}
        )
        if _sha256_text(gold) != schedule.get("gold_sha256"):
            raise DatasetReleaseValidationError(f"{label} gold hash drift")
        direction_counts[parsed_decision[0]] += 1
        repeat_indexes.setdefault(str(sample_id), []).append(repeat_index)
        if str(sample_id) in repeat_totals and repeat_totals[str(sample_id)] != (
            repeat_total
        ):
            raise DatasetReleaseValidationError(f"{label} repeat total drift")
        repeat_totals[str(sample_id)] = repeat_total

    if direction_counts != CHK4_HIER_BALANCED_DIRECTION_COUNTS:
        raise DatasetReleaseValidationError("hier-balanced direction counts drift")
    if set(repeat_indexes) != set(unique_by_id):
        raise DatasetReleaseValidationError("hier-balanced unique coverage drift")
    for sample_id, indexes in repeat_indexes.items():
        if indexes != list(range(repeat_totals[sample_id])):
            raise DatasetReleaseValidationError(
                f"hier-balanced repeat sequence drift: {sample_id}"
            )
    for step in range(CHK4_HIER_BALANCED_STEPS):
        start = step * CHK4_HIER_BALANCED_EFFECTIVE_BATCH
        window = schedule_rows[start : start + CHK4_HIER_BALANCED_EFFECTIVE_BATCH]
        counts = {"hold": 0, "hike": 0, "cut": 0}
        for row in window:
            counts[str(row["direction"])] += 1
        if (
            counts != CHK4_HIER_BALANCED_WINDOW_COUNTS
            or len({str(row["source_sample_id"]) for row in window})
            != CHK4_HIER_BALANCED_EFFECTIVE_BATCH
        ):
            raise DatasetReleaseValidationError(
                f"hier-balanced optimizer window {step} drift"
            )

    sampler = _required_mapping(
        root.get("sampler_contract"), label="hier-balanced sampler_contract"
    )
    expected_sampler = {
        "type": CHK4_HIER_BALANCED_SAMPLER_TYPE,
        "seed": 20260811,
        "schedule_path": "manifests/sampler_schedule.jsonl",
        "schedule_sha256": normalized_files["manifests/sampler_schedule.jsonl"][
            "sha256"
        ],
        "schedule_rows": CHK4_HIER_BALANCED_ROWS,
        "train_path": "decision_sft/train.jsonl",
        "train_sha256": normalized_files["decision_sft/train.jsonl"]["sha256"],
        "train_rows": CHK4_HIER_BALANCED_ROWS,
        "effective_batch_size": CHK4_HIER_BALANCED_EFFECTIVE_BATCH,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "world_size": 1,
        "optimizer_steps": CHK4_HIER_BALANCED_STEPS,
        "shuffle_dataset": False,
        "direction_counts": CHK4_HIER_BALANCED_DIRECTION_COUNTS,
        "per_optimizer_window": CHK4_HIER_BALANCED_WINDOW_COUNTS,
        "order_is_authoritative": True,
        "required_sampler": "fixed_sequential_schedule_index_v1",
        "secondary_shuffle_forbidden": True,
    }
    if dict(sampler) != expected_sampler:
        raise DatasetReleaseValidationError("hier-balanced sampler contract drift")

    token_contract = _required_mapping(
        root.get("token_contract"), label="hier-balanced token_contract"
    )
    parent_tokenizer = _required_mapping(
        parent_verified.get("tokenizer_binding"),
        label="hier-balanced parent tokenizer binding",
    )
    if (
        token_contract.get("tokenizer_files") != parent_tokenizer.get("files")
        or token_contract.get("tokenizer_bundle_sha256")
        != parent_tokenizer.get("bundle_sha256")
        or token_contract.get("single_bos") is not True
        or token_contract.get("final_eos") is not True
        or token_contract.get("completion_mask_covers_reasoning_boundary_answer_eos")
        is not True
        or token_contract.get("max_prompt_length") != 2560
        or token_contract.get("max_completion_length") != 512
        or token_contract.get("sft_max_length") != 3072
        or token_contract.get("truncation") is not False
    ):
        raise DatasetReleaseValidationError("hier-balanced token contract drift")

    implementation = _required_mapping(
        root.get("implementation"), label="hier-balanced implementation"
    )
    for prefix in ("materializer", "parent_publisher", "sft_prompt_renderer"):
        relative = implementation.get(f"{prefix}_snapshot")
        digest = implementation.get(f"{prefix}_snapshot_sha256")
        if (
            not isinstance(relative, str)
            or sha256_file(
                _resolve_release_member(
                    release_root,
                    relative,
                    label=f"hier-balanced {prefix} snapshot",
                )
            )
            != digest
        ):
            raise DatasetReleaseValidationError(
                f"hier-balanced {prefix} implementation drift"
            )

    handoff_record = _required_mapping(
        root.get("handoff"), label="hier-balanced handoff binding"
    )
    handoff = _read_json_object(
        release_root / "handoff.json", label="hier-balanced handoff"
    )
    unsigned_handoff = dict(handoff)
    unsigned_handoff.pop("release_manifest_sha256", None)
    if handoff_record != {
        "path": "handoff.json",
        "schema_version": CHK4_HIER_BALANCED_HANDOFF_SCHEMA,
        "unsigned_payload_sha256": _sha256_text(_canonical_json(unsigned_handoff)),
    } or not (
        handoff.get("schema_version") == CHK4_HIER_BALANCED_HANDOFF_SCHEMA
        and handoff.get("release_id") == CHK4_HIER_BALANCED_RELEASE_ID
        and handoff.get("quality_status") == "passed"
        and handoff.get("immutable") is True
        and handoff.get("training_ready") is True
        and handoff.get("dataset_role") == CHK4_HIER_BALANCED_DATASET_ROLE
        and handoff.get("release_manifest") == "release_manifest.json"
        and handoff.get("release_manifest_sha256") == expected_sha
        and handoff.get("dataset_path") == "decision_sft"
        and handoff.get("sampler_schedule") == "manifests/sampler_schedule.jsonl"
        and handoff.get("test_is_sealed_evaluation_only") is True
    ):
        raise DatasetReleaseValidationError("hier-balanced handoff drift")

    normalized_sampler = dict(expected_sampler)
    normalized_sampler["schedule_path"] = str(schedule_path)
    normalized_sampler["train_path"] = str(train_path)
    return {
        "schema_version": "chk4-hier-balanced-runtime-binding-v1",
        "release_id": CHK4_HIER_BALANCED_RELEASE_ID,
        "dataset_role": CHK4_HIER_BALANCED_DATASET_ROLE,
        "release_manifest_path": str(manifest),
        "release_manifest_sha256": expected_sha,
        "split_files": {
            "train": train_path,
            "validation": release_root / "decision_sft/validation.jsonl",
        },
        "test_verified_but_not_loaded": True,
        "system_prompt_sha256": CHK4_STUDENT_SYSTEM_PROMPT_SHA256,
        "tokenizer_binding": dict(parent_tokenizer),
        "sampler_contract": normalized_sampler,
    }


def verify_chk4_pre2009_augmented_release(
    *,
    dataset_dir: str | Path,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    dataset_role: str,
    system_prompt: str | None,
    model_path: str | Path,
) -> dict[str, Any]:
    """Deeply replay and bind the train-only pre-2009 augmented release.

    The release publisher owns the full lineage replay, including provider
    content to SFT reconstruction.  Runtime calls that verifier with an
    externally pinned manifest digest, then independently reuses the sealed
    core-v3 verifier to bind the current model's tokenizer.
    """

    if dataset_role not in CHK4_PRE2009_ROLES:
        raise DatasetReleaseValidationError(
            "pre-2009 augmented release requires role "
            f"{CHK4_PRE2009_SFT_ROLE} or {CHK4_PRE2009_GRPO_ROLE}"
        )
    expected_sha = _required_sha256(
        expected_manifest_sha256,
        label="dataset_chk4_release_manifest_sha256",
    )
    if system_prompt != CHK4_STUDENT_SYSTEM_PROMPT:
        raise DatasetReleaseValidationError(
            "chk4 system_prompt does not match the pre-2009 augmented release"
        )
    dataset = _canonical_path(dataset_dir, label="dataset_name", directory=True)
    manifest = _canonical_path(
        manifest_path, label="dataset_chk4_release_manifest", directory=False
    )
    release_root = manifest.parent
    physical_role = CHK4_PRE2009_PHYSICAL_ROLE[dataset_role]
    if dataset != release_root / physical_role:
        raise DatasetReleaseValidationError(
            "pre-2009 dataset_name does not match its logical chk4 role"
        )
    if release_root.name != CHK4_PRE2009_AUGMENTED_RELEASE_ID:
        raise DatasetReleaseValidationError("pre-2009 release directory drift")
    if sha256_file(manifest) != expected_sha:
        raise DatasetReleaseValidationError(
            "pre-2009 release manifest SHA-256 disagrees with the training config"
        )
    root_manifest = _read_json_object(manifest, label="pre-2009 release manifest")
    if (
        root_manifest.get("schema_version") != CHK4_PRE2009_AUGMENTED_RELEASE_SCHEMA
        or root_manifest.get("release_id") != CHK4_PRE2009_AUGMENTED_RELEASE_ID
    ):
        raise DatasetReleaseValidationError("pre-2009 release identity drift")

    try:
        from jobs.retrain_v2.materialize_chk4_pre2009_augmented_release import (
            Pre2009ReleaseError,
            verify_release,
        )
    except ImportError as exc:
        raise DatasetReleaseValidationError(
            f"pre-2009 augmented release verifier import failed: {exc}"
        ) from exc
    try:
        verified = verify_release(
            release_root,
            expected_manifest_sha256=expected_sha,
        )
    except (OSError, ValueError, Pre2009ReleaseError) as exc:
        raise DatasetReleaseValidationError(
            f"pre-2009 augmented release replay failed: {exc}"
        ) from exc

    parents = _required_mapping(
        verified.get("parent_releases"), label="pre-2009 parent releases"
    )
    core_record = _required_mapping(
        parents.get("core_v3"), label="pre-2009 core-v3 parent"
    )
    core_path_value = core_record.get("path")
    if not isinstance(core_path_value, str) or not core_path_value:
        raise DatasetReleaseValidationError("pre-2009 core-v3 path is missing")
    core_path = Path(core_path_value)
    if not core_path.is_absolute():
        core_path = Path(__file__).resolve().parents[3] / core_path
    core_root = _canonical_path(
        core_path, label="pre-2009 core-v3 parent", directory=True
    )
    core_manifest_sha = _required_sha256(
        core_record.get("manifest_sha256"),
        label="pre-2009 core-v3 manifest SHA",
    )
    core_verified = verify_chk4_decision_release(
        dataset_dir=core_root / "decision_sft",
        manifest_path=core_root / "release_manifest.json",
        expected_manifest_sha256=core_manifest_sha,
        dataset_role="decision_sft",
        system_prompt=system_prompt,
        model_path=model_path,
    )

    sampler = _required_mapping(
        verified.get("sampler_contract"), label="pre-2009 sampler contract"
    )
    if sampler.get("type") != CHK4_PRE2009_SAMPLER_TYPE:
        raise DatasetReleaseValidationError("pre-2009 sampler type drift")
    schedule_path = _resolve_release_member(
        release_root,
        sampler.get("schedule_path"),
        label="pre-2009 sampler schedule",
    )
    train_path = _resolve_release_member(
        release_root,
        sampler.get("train_path"),
        label="pre-2009 sampler train file",
    )
    normalized_sampler = dict(sampler)
    normalized_sampler["schedule_path"] = str(schedule_path)
    normalized_sampler["train_path"] = str(train_path)
    split_files = _required_mapping(
        verified.get("verified_split_files"),
        label="pre-2009 verified split files",
    )
    physical_splits = _required_mapping(
        split_files.get(physical_role),
        label=f"pre-2009 {physical_role} split files",
    )
    return {
        "schema_version": "chk4-pre2009-augmented-runtime-binding-v1",
        "release_id": CHK4_PRE2009_AUGMENTED_RELEASE_ID,
        "dataset_role": dataset_role,
        "physical_dataset_role": physical_role,
        "release_manifest_path": str(manifest),
        "release_manifest_sha256": expected_sha,
        "split_files": {
            "train": Path(str(physical_splits["train"])).resolve(),
            "validation": Path(str(physical_splits["validation"])).resolve(),
        },
        "test_verified_but_not_loaded": True,
        "system_prompt_sha256": CHK4_STUDENT_SYSTEM_PROMPT_SHA256,
        "tokenizer_binding": dict(core_verified["tokenizer_binding"]),
        "sampler_contract": normalized_sampler,
    }


def verify_chk4_pre2009_correction_release(
    *,
    dataset_dir: str | Path,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    dataset_role: str,
    system_prompt: str | None,
    model_path: str | Path,
) -> dict[str, Any]:
    """Replay and bind the independent 48-row correction-SFT child.

    The child verifier is additive: it delegates full lineage replay to its
    own versioned materializer, while the existing sealed parent verifier
    continues to own tokenizer binding.  The externally supplied child
    manifest digest is mandatory; a self-reported digest is never trusted.
    """

    if dataset_role != CHK4_PRE2009_CORRECTION_SFT_ROLE:
        raise DatasetReleaseValidationError(
            "pre-2009 correction release requires dataset_chk4_role="
            f"{CHK4_PRE2009_CORRECTION_SFT_ROLE}"
        )
    expected_sha = _required_sha256(
        expected_manifest_sha256,
        label="dataset_chk4_release_manifest_sha256",
    )
    if system_prompt != CHK4_STUDENT_SYSTEM_PROMPT:
        raise DatasetReleaseValidationError(
            "chk4 system_prompt does not match the correction release"
        )
    dataset = _canonical_path(dataset_dir, label="dataset_name", directory=True)
    manifest = _canonical_path(
        manifest_path, label="dataset_chk4_release_manifest", directory=False
    )
    release_root = manifest.parent
    if dataset != release_root / "decision_sft":
        raise DatasetReleaseValidationError(
            "correction dataset_name must be the child decision_sft directory"
        )
    if release_root.name != CHK4_PRE2009_CORRECTION_RELEASE_ID:
        raise DatasetReleaseValidationError("correction release directory drift")
    if sha256_file(manifest) != expected_sha:
        raise DatasetReleaseValidationError(
            "correction manifest SHA-256 disagrees with the training config"
        )
    root_manifest = _read_json_object(
        manifest, label="pre-2009 correction release manifest"
    )
    if (
        root_manifest.get("schema_version")
        != CHK4_PRE2009_CORRECTION_RELEASE_SCHEMA
        or root_manifest.get("release_id")
        != CHK4_PRE2009_CORRECTION_RELEASE_ID
        or root_manifest.get("dataset_role")
        != CHK4_PRE2009_CORRECTION_SFT_ROLE
    ):
        raise DatasetReleaseValidationError("correction release identity drift")

    try:
        from jobs.retrain_v2.materialize_chk4_pre2009_correction_release import (
            CorrectionReleaseError,
            verify_release,
        )
    except ImportError as exc:
        raise DatasetReleaseValidationError(
            f"correction release verifier import failed: {exc}"
        ) from exc
    try:
        verified = verify_release(
            release_root,
            expected_manifest_sha256=expected_sha,
        )
    except (OSError, ValueError, CorrectionReleaseError) as exc:
        raise DatasetReleaseValidationError(
            f"correction release replay failed: {exc}"
        ) from exc

    parent_record = _required_mapping(
        verified.get("parent_release"), label="correction parent release"
    )
    parent_path_value = parent_record.get("path")
    if not isinstance(parent_path_value, str) or not parent_path_value:
        raise DatasetReleaseValidationError("correction parent path is missing")
    parent_path = Path(parent_path_value)
    if not parent_path.is_absolute():
        parent_path = Path(__file__).resolve().parents[3] / parent_path
    parent_root = _canonical_path(
        parent_path, label="correction parent release", directory=True
    )
    parent_manifest_sha = _required_sha256(
        parent_record.get("manifest_sha256"),
        label="correction parent manifest SHA",
    )
    parent_verified = verify_chk4_pre2009_augmented_release(
        dataset_dir=parent_root / "decision_sft",
        manifest_path=parent_root / "release_manifest.json",
        expected_manifest_sha256=parent_manifest_sha,
        dataset_role=CHK4_PRE2009_SFT_ROLE,
        system_prompt=system_prompt,
        model_path=model_path,
    )

    role_contract = _required_mapping(
        verified.get("training_role"), label="correction training role"
    )
    expected_role_contract = {
        "dataset_path": "decision_sft",
        "completion_only_loss": True,
        "max_length": 3072,
        "parent_role": "selected_pre2009_cp38_exact_merged",
        "sampler_contract": "sampler_contract",
    }
    if dict(role_contract) != expected_role_contract:
        raise DatasetReleaseValidationError(
            "correction training role contract drift"
        )
    sampler = _required_mapping(
        verified.get("sampler_contract"), label="correction sampler contract"
    )
    if sampler.get("type") != CHK4_PRE2009_CORRECTION_SAMPLER_TYPE:
        raise DatasetReleaseValidationError("correction sampler type drift")
    schedule_path = _resolve_release_member(
        release_root,
        sampler.get("schedule_path"),
        label="correction sampler schedule",
    )
    train_path = _resolve_release_member(
        release_root,
        sampler.get("train_path"),
        label="correction sampler train file",
    )
    validation_path = _canonical_path(
        release_root / "decision_sft/validation.jsonl",
        label="correction validation split",
        directory=False,
    )
    normalized_sampler = dict(sampler)
    normalized_sampler["schedule_path"] = str(schedule_path)
    normalized_sampler["train_path"] = str(train_path)
    return {
        "schema_version": "chk4-pre2009-correction-runtime-binding-v1",
        "release_id": CHK4_PRE2009_CORRECTION_RELEASE_ID,
        "dataset_role": CHK4_PRE2009_CORRECTION_SFT_ROLE,
        "physical_dataset_role": "decision_sft",
        "release_manifest_path": str(manifest),
        "release_manifest_sha256": expected_sha,
        "split_files": {
            "train": train_path,
            "validation": validation_path,
        },
        "test_verified_but_not_loaded": True,
        "system_prompt_sha256": CHK4_STUDENT_SYSTEM_PROMPT_SHA256,
        "tokenizer_binding": dict(parent_verified["tokenizer_binding"]),
        "sampler_contract": normalized_sampler,
        "heldout_contract": dict(
            _required_mapping(
                verified.get("heldout_contract"),
                label="correction heldout contract",
            )
        ),
    }


def verify_chk4_release_for_role(
    *,
    dataset_dir: str | Path,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    dataset_role: str,
    system_prompt: str | None,
    model_path: str | Path,
) -> dict[str, Any]:
    """Route one logical chk4 role to its versioned fail-closed verifier."""

    verifier = verify_chk4_decision_release
    if dataset_role == CHK4_PRE2009_CORRECTION_SFT_ROLE:
        verifier = verify_chk4_pre2009_correction_release
    elif dataset_role == CHK4_HIER_BALANCED_DATASET_ROLE:
        verifier = verify_chk4_hier_balanced_release
    elif dataset_role in CHK4_PRE2009_ROLES:
        verifier = verify_chk4_pre2009_augmented_release
    return verifier(
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        dataset_role=dataset_role,
        system_prompt=system_prompt,
        model_path=model_path,
    )


def verify_clean_sft_release(
    *,
    dataset_dir: str | Path,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify an already-audited release without rerunning semantic judges."""

    expected_sha = _required_sha256(
        expected_manifest_sha256, label="dataset_release_manifest_sha256"
    )
    dataset = _canonical_path(dataset_dir, label="dataset_name", directory=True)
    manifest = _canonical_path(
        manifest_path, label="dataset_release_manifest", directory=False
    )
    release_root = manifest.parent
    if dataset.parent != release_root:
        raise DatasetReleaseValidationError(
            "dataset_name and dataset_release_manifest must belong to the same release"
        )
    observed_manifest_sha = sha256_file(manifest)
    if observed_manifest_sha != expected_sha:
        raise DatasetReleaseValidationError(
            "dataset release manifest SHA-256 disagrees with the training config"
        )
    root = _read_json_object(manifest, label="dataset release manifest")
    if root.get("schema_version") != CLEAN_SFT_RELEASE_SCHEMA:
        raise DatasetReleaseValidationError(
            "dataset release schema is not clean-SFT v2"
        )
    if root.get("quality_status") != "passed" or root.get("immutable") is not True:
        raise DatasetReleaseValidationError(
            "dataset release must be immutable with quality_status=passed"
        )

    split_counts = _required_mapping(root.get("split_counts"), label="split_counts")
    split_files = _required_mapping(root.get("split_files"), label="split_files")
    if set(split_counts) != set(_SPLITS) or set(split_files) != set(_SPLITS):
        raise DatasetReleaseValidationError(
            "split_counts and split_files must contain train/eval/test exactly"
        )
    total_rows = 0
    for split in _SPLITS:
        info = _required_mapping(split_files[split], label=f"split_files.{split}")
        path = _resolve_release_member(
            release_root, info.get("path"), label=f"split_files.{split}.path"
        )
        expected_path = dataset / f"{split}.jsonl"
        if path != expected_path:
            raise DatasetReleaseValidationError(
                f"split_files.{split}.path does not bind dataset_name"
            )
        expected_rows = _required_count(
            split_counts[split], label=f"split_counts.{split}"
        )
        if (
            _required_count(info.get("rows"), label=f"split_files.{split}.rows")
            != expected_rows
        ):
            raise DatasetReleaseValidationError(
                f"split_files.{split}.rows disagrees with split_counts"
            )
        if sha256_file(path) != _required_sha256(
            info.get("sha256"), label=f"split_files.{split}.sha256"
        ):
            raise DatasetReleaseValidationError(f"{split} split SHA-256 mismatch")
        if len(_read_jsonl_objects(path, label=f"{split} split")) != expected_rows:
            raise DatasetReleaseValidationError(f"{split} split row-count mismatch")
        total_rows += expected_rows

    repair = _required_mapping(root.get("repair_manifest"), label="repair_manifest")
    repair_path = _resolve_release_member(
        release_root, repair.get("path"), label="repair_manifest.path"
    )
    if sha256_file(repair_path) != _required_sha256(
        repair.get("sha256"), label="repair_manifest.sha256"
    ):
        raise DatasetReleaseValidationError("repair manifest SHA-256 mismatch")
    if _required_count(repair.get("rows"), label="repair_manifest.rows") != total_rows:
        raise DatasetReleaseValidationError("repair manifest row count is incomplete")
    repair_rows = _read_jsonl_objects(repair_path, label="repair manifest")
    if len(repair_rows) != total_rows:
        raise DatasetReleaseValidationError(
            "repair manifest physical row count is incomplete"
        )

    changed_rows = _required_count(root.get("changed_rows"), label="changed_rows")
    repair_sample_ids: set[str] = set()
    expected_changed: dict[str, tuple[str, str, str]] = {}
    content_binding: list[dict[str, Any]] = []
    for number, row in enumerate(repair_rows, 1):
        sample_id = row.get("sample_id")
        split = row.get("split")
        line_number = row.get("source_line_number")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in repair_sample_ids
            or split not in _SPLITS
            or isinstance(line_number, bool)
            or not isinstance(line_number, int)
            or line_number < 1
        ):
            raise DatasetReleaseValidationError(
                f"repair manifest row {number} has invalid identity"
            )
        repair_sample_ids.add(sample_id)
        hashes = {
            name: _required_sha256(
                row.get(name), label=f"repair manifest row {number}.{name}"
            )
            for name in (
                "prompt_sha256",
                "provided_data_sha256",
                "old_response_sha256",
                "new_response_sha256",
            )
        }
        content_binding.append(
            {
                "sample_id": sample_id,
                "split": split,
                "line_number": line_number,
                "prompt_sha256": hashes["prompt_sha256"],
                "provided_data_sha256": hashes["provided_data_sha256"],
                "response_sha256": hashes["new_response_sha256"],
            }
        )
        if hashes["old_response_sha256"] != hashes["new_response_sha256"]:
            expected_changed[sample_id] = (
                str(split),
                hashes["new_response_sha256"],
                hashes["provided_data_sha256"],
            )
    if len(expected_changed) != changed_rows:
        raise DatasetReleaseValidationError(
            "repair manifest changed-row count mismatch"
        )
    declared_changed_ids = _required_sha256(
        root.get("changed_sample_ids_sha256"), label="changed_sample_ids_sha256"
    )
    if declared_changed_ids != _sha256_text(_canonical_json(sorted(expected_changed))):
        raise DatasetReleaseValidationError("changed sample-ID binding mismatch")

    semantic = _required_mapping(root.get("semantic_audit"), label="semantic_audit")
    if semantic.get("schema_version") != CLEAN_SFT_AUDIT_SCHEMA:
        raise DatasetReleaseValidationError(
            "semantic audit schema is not source-audit v1"
        )
    summary_info = _required_mapping(
        semantic.get("summary"), label="semantic_audit.summary"
    )
    summary_path = _resolve_release_member(
        release_root,
        summary_info.get("path"),
        label="semantic_audit.summary.path",
    )
    if sha256_file(summary_path) != _required_sha256(
        summary_info.get("sha256"), label="semantic_audit.summary.sha256"
    ):
        raise DatasetReleaseValidationError("semantic audit summary SHA-256 mismatch")
    summary = _read_json_object(summary_path, label="semantic audit summary")
    if (
        summary.get("schema_version") != CLEAN_SFT_AUDIT_SCHEMA
        or summary.get("status") != "passed"
    ):
        raise DatasetReleaseValidationError("semantic audit summary did not pass")
    counts = _required_mapping(summary.get("counts"), label="semantic audit counts")
    if any(
        _required_count(counts.get(key), label=f"semantic audit counts.{key}")
        != expected
        for key, expected in (
            ("expected", changed_rows),
            ("completed", changed_rows),
            ("passed", changed_rows),
            ("failed", 0),
            ("judge_errors", 0),
            ("blocking_violations", 0),
        )
    ):
        raise DatasetReleaseValidationError(
            "semantic audit counts are not fully passed"
        )
    if summary.get("repair_manifest_sha256") != repair.get("sha256"):
        raise DatasetReleaseValidationError(
            "semantic audit repair-manifest binding mismatch"
        )
    if summary.get("errors") != []:
        raise DatasetReleaseValidationError("semantic audit contains judge errors")
    judge = _required_mapping(summary.get("judge"), label="semantic audit judge")
    health = _required_mapping(judge.get("health"), label="semantic audit judge.health")
    if (
        judge.get("model") != _JUDGE_MODEL
        or health.get("model") != _JUDGE_MODEL
        or Path(str(health.get("loaded_model_root") or "")).name != _JUDGE_MODEL
        or health.get("status") != "ready"
        or health.get("tokenizer_parity") is not True
        or health.get("weight_attested") is not True
    ):
        raise DatasetReleaseValidationError(
            "semantic audit judge health is not attested"
        )

    rows_info = _required_mapping(semantic.get("rows"), label="semantic_audit.rows")
    rows_path = _resolve_release_member(
        release_root, rows_info.get("path"), label="semantic_audit.rows.path"
    )
    rows_sha = _required_sha256(
        rows_info.get("sha256"), label="semantic_audit.rows.sha256"
    )
    if sha256_file(rows_path) != rows_sha:
        raise DatasetReleaseValidationError("semantic audit rows SHA-256 mismatch")
    if (
        _required_count(rows_info.get("rows"), label="semantic_audit.rows.rows")
        != changed_rows
        or len(_read_jsonl_objects(rows_path, label="semantic audit rows"))
        != changed_rows
    ):
        raise DatasetReleaseValidationError("semantic audit row count is incomplete")
    summary_rows = _required_mapping(
        summary.get("row_audit"), label="semantic audit summary.row_audit"
    )
    if (
        _required_count(summary_rows.get("rows"), label="summary.row_audit.rows")
        != changed_rows
        or _required_sha256(
            summary_rows.get("sha256"), label="summary.row_audit.sha256"
        )
        != rows_sha
    ):
        raise DatasetReleaseValidationError(
            "semantic audit summary row binding mismatch"
        )

    semantic_rows = _read_jsonl_objects(rows_path, label="semantic audit rows")
    observed_changed: dict[str, tuple[str, str, str]] = {}
    observed_pairs: set[tuple[str, str]] = set()
    for number, row in enumerate(semantic_rows, 1):
        sample_id = row.get("sample_id")
        split = row.get("split")
        if (
            row.get("schema_version") != CLEAN_SFT_AUDIT_SCHEMA
            or row.get("status") != "passed"
            or not isinstance(sample_id, str)
            or not sample_id
            or sample_id in observed_changed
            or split not in _SPLITS
        ):
            raise DatasetReleaseValidationError(
                f"semantic audit row {number} identity/status is invalid"
            )
        candidate_sha = _required_sha256(
            row.get("candidate_sha256"),
            label=f"semantic audit row {number}.candidate_sha256",
        )
        evidence_sha = _required_sha256(
            row.get("evidence_sha256"),
            label=f"semantic audit row {number}.evidence_sha256",
        )
        pair = (sample_id, candidate_sha)
        if pair in observed_pairs:
            raise DatasetReleaseValidationError(
                "duplicate semantic sample/candidate binding"
            )
        observed_pairs.add(pair)
        blocking = row.get("blocking_violations")
        validated = row.get("validated_violations")
        if blocking != [] or not isinstance(validated, list):
            raise DatasetReleaseValidationError(
                f"semantic audit row {number} contains blocking violations"
            )
        if any(
            isinstance(violation, Mapping) and violation.get("kind") in _BLOCKING_KINDS
            for violation in validated
        ):
            raise DatasetReleaseValidationError(
                f"semantic audit row {number} validated violation is blocking"
            )
        rubric = row.get("rubric")
        if (
            not isinstance(rubric, Mapping)
            or set(rubric) != _RUBRIC_KEYS
            or not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= 4
                for value in rubric.values()
            )
        ):
            raise DatasetReleaseValidationError(
                f"semantic audit row {number} rubric is invalid"
            )
        attempts = row.get("judge_attempts")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise DatasetReleaseValidationError(
                f"semantic audit row {number} judge attempts are invalid"
            )
        _required_sha256(
            row.get("judge_raw_sha256"),
            label=f"semantic audit row {number}.judge_raw_sha256",
        )
        observed_changed[sample_id] = (str(split), candidate_sha, evidence_sha)
    if observed_changed != expected_changed:
        raise DatasetReleaseValidationError(
            "semantic audit rows are not bound to changed repair rows"
        )

    deterministic = _required_mapping(
        root.get("deterministic_validation"), label="deterministic_validation"
    )
    if deterministic.get("schema_version") != CLEAN_SFT_VALIDATION_SCHEMA:
        raise DatasetReleaseValidationError("deterministic validation schema mismatch")
    report_info = _required_mapping(
        deterministic.get("report"), label="deterministic_validation.report"
    )
    validation_path = _resolve_release_member(
        release_root,
        report_info.get("path"),
        label="deterministic_validation.report.path",
    )
    if sha256_file(validation_path) != _required_sha256(
        report_info.get("sha256"), label="deterministic_validation.report.sha256"
    ):
        raise DatasetReleaseValidationError(
            "deterministic validation report SHA-256 mismatch"
        )
    validation = _read_json_object(
        validation_path, label="deterministic validation report"
    )
    if (
        validation.get("schema_version") != CLEAN_SFT_VALIDATION_SCHEMA
        or validation.get("status") != "passed"
    ):
        raise DatasetReleaseValidationError("deterministic validation did not pass")
    claimed_validation_sha = _required_sha256(
        validation.get("validation_sha256"), label="validation_sha256"
    )
    validation_digest_payload = dict(validation)
    validation_digest_payload.pop("validation_sha256", None)
    if claimed_validation_sha != _sha256_text(
        _canonical_json(validation_digest_payload)
    ):
        raise DatasetReleaseValidationError(
            "deterministic validation self-digest mismatch"
        )
    if (
        deterministic.get("validation_sha256") != claimed_validation_sha
        or deterministic.get("max_length") != 4608
        or deterministic.get("tokenizer_bundle_sha256")
        != _required_sha256(
            _required_mapping(
                _required_mapping(
                    validation.get("inputs"), label="validation inputs"
                ).get("tokenizer"),
                label="validation tokenizer",
            ).get("tokenizer_bundle_sha256"),
            label="validation tokenizer bundle",
        )
    ):
        raise DatasetReleaseValidationError(
            "deterministic validation manifest binding mismatch"
        )
    contracts = _required_mapping(
        validation.get("contracts"), label="validation contracts"
    )
    if (
        contracts.get("data") != CLEAN_SFT_DATA_CONTRACT
        or contracts.get("tokenization") != CLEAN_SFT_TOKEN_CONTRACT
    ):
        raise DatasetReleaseValidationError(
            "deterministic validation contract mismatch"
        )
    configuration = _required_mapping(
        validation.get("configuration"), label="validation configuration"
    )
    if (
        configuration.get("expected_split_counts") != dict(split_counts)
        or configuration.get("max_length") != 4608
        or configuration.get("reasoning_token_range") != [512, 2400]
        or configuration.get("answer_token_range") != [16, 512]
    ):
        raise DatasetReleaseValidationError(
            "deterministic validation configuration mismatch"
        )
    validation_inputs = _required_mapping(
        validation.get("inputs"), label="validation inputs"
    )
    clean_input = _required_mapping(
        validation_inputs.get("clean_release"), label="validation clean_release"
    )
    candidate_info = _required_mapping(
        root.get("candidate_manifest"), label="candidate_manifest"
    )
    candidate_path = candidate_info.get("path")
    candidate_sha = _required_sha256(
        candidate_info.get("sha256"), label="candidate_manifest.sha256"
    )
    if not isinstance(candidate_path, str) or not candidate_path:
        raise DatasetReleaseValidationError("candidate_manifest.path is invalid")
    if clean_input.get("release_id") != Path(candidate_path).parent.name:
        raise DatasetReleaseValidationError(
            "deterministic validation candidate identity mismatch"
        )
    clean_manifests = clean_input.get("manifests")
    if (
        not isinstance(clean_manifests, list)
        or sum(
            isinstance(item, Mapping)
            and item.get("name") == "candidate_manifest.json"
            and item.get("sha256") == candidate_sha
            for item in clean_manifests
        )
        != 1
    ):
        raise DatasetReleaseValidationError(
            "deterministic validation candidate hash mismatch"
        )
    clean_splits = _required_mapping(
        clean_input.get("splits"), label="validation clean splits"
    )
    for split in _SPLITS:
        split_input = _required_mapping(
            clean_splits.get(split), label=f"validation clean split {split}"
        )
        if split_input.get("sha256") != split_files[split].get("sha256"):
            raise DatasetReleaseValidationError(
                f"deterministic validation {split} split binding mismatch"
            )
    validation_repair = _required_mapping(
        validation_inputs.get("repair_manifest"), label="validation repair_manifest"
    )
    validation_tokenizer = _required_mapping(
        validation_inputs.get("tokenizer"), label="validation tokenizer"
    )
    tokenizer_files = validation_tokenizer.get("files")
    if not isinstance(tokenizer_files, list) or not tokenizer_files:
        raise DatasetReleaseValidationError("validation tokenizer files are missing")
    for number, tokenizer_file in enumerate(tokenizer_files, 1):
        if (
            not isinstance(tokenizer_file, Mapping)
            or not isinstance(tokenizer_file.get("name"), str)
            or not tokenizer_file.get("name")
            or isinstance(tokenizer_file.get("bytes"), bool)
            or not isinstance(tokenizer_file.get("bytes"), int)
            or tokenizer_file.get("bytes") < 0
        ):
            raise DatasetReleaseValidationError(
                f"validation tokenizer file {number} is invalid"
            )
        _required_sha256(
            tokenizer_file.get("sha256"),
            label=f"validation tokenizer file {number}.sha256",
        )
    tokenizer_bundle_sha = _required_sha256(
        validation_tokenizer.get("tokenizer_bundle_sha256"),
        label="validation tokenizer bundle",
    )
    if tokenizer_bundle_sha != _sha256_text(_canonical_json(tokenizer_files)):
        raise DatasetReleaseValidationError(
            "validation tokenizer bundle digest mismatch"
        )
    if (
        validation_repair.get("sha256") != repair.get("sha256")
        or validation_repair.get("rows") != total_rows
        or validation.get("content_binding_sha256")
        != _sha256_text(_canonical_json(content_binding))
    ):
        raise DatasetReleaseValidationError(
            "deterministic validation content binding mismatch"
        )
    validation_counts = _required_mapping(
        validation.get("counts"), label="validation counts"
    )
    for key, expected in (
        ("expected_rows", total_rows),
        ("observed_rows", total_rows),
        ("valid_rows", total_rows),
        ("prompt_hashes_unchanged", total_rows),
        ("provided_data_hashes_unchanged", total_rows),
        ("order_unchanged", total_rows),
        ("truncated_rows", 0),
        ("issue_rows_or_groups", 0),
    ):
        if validation_counts.get(key) != expected:
            raise DatasetReleaseValidationError(
                f"deterministic validation count mismatch: {key}"
            )
    if validation_counts.get("split_counts") != dict(split_counts):
        raise DatasetReleaseValidationError(
            "deterministic validation split count mismatch"
        )
    gates = _required_mapping(validation.get("quality_gates"), label="validation gates")
    if not gates or not all(value is True for value in gates.values()):
        raise DatasetReleaseValidationError("deterministic validation gate failed")
    if validation.get("issues") != []:
        raise DatasetReleaseValidationError("deterministic validation contains issues")
    statistics = _required_mapping(
        validation.get("token_statistics"), label="validation token_statistics"
    )
    max_total_tokens = statistics.get("max_total_tokens")
    if (
        isinstance(max_total_tokens, bool)
        or not isinstance(max_total_tokens, int)
        or not 0 < max_total_tokens <= 4608
    ):
        raise DatasetReleaseValidationError(
            "deterministic validation max token count is invalid"
        )
    return dict(root)


def verify_standalone_chk3_direct_sft_release(
    *,
    dataset_dir: str | Path,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    training_scope: str,
) -> dict[str, Any]:
    """Verify a sealed standalone Minutes-SFT release for a direct chk1 branch.

    This contract is intentionally separate from the canonical retrain-v2 DAG.
    Passing it proves the bytes consumed by SFT, but never makes the resulting
    adapter a canonical chk3 artifact or authorizes downstream promotion.
    """

    if training_scope != STANDALONE_CHK3_DIRECT_SCOPE:
        raise DatasetReleaseValidationError(
            "standalone chk3 training scope must be the explicit non-promotable "
            "chk1-to-chk3 direct-SFT scope"
        )
    expected_sha = _required_sha256(
        expected_manifest_sha256,
        label="dataset_standalone_chk3_release_manifest_sha256",
    )
    dataset = _canonical_path(dataset_dir, label="dataset_name", directory=True)
    manifest = _canonical_path(
        manifest_path,
        label="dataset_standalone_chk3_release_manifest",
        directory=False,
    )
    release_root = manifest.parent
    if dataset.parent != release_root or dataset.name != "minutes_alignment":
        raise DatasetReleaseValidationError(
            "dataset_name must be the minutes_alignment directory beside the "
            "standalone chk3 release manifest"
        )
    if sha256_file(manifest) != expected_sha:
        raise DatasetReleaseValidationError(
            "standalone chk3 release manifest SHA-256 disagrees with the training config"
        )

    root = _read_json_object(manifest, label="standalone chk3 release manifest")
    if root.get("schema_version") != STANDALONE_CHK3_RELEASE_SCHEMA:
        raise DatasetReleaseValidationError(
            "standalone chk3 release schema is not chk3 Minutes release v1"
        )
    if root.get("dataset_role") != STANDALONE_CHK3_DATASET_ROLE:
        raise DatasetReleaseValidationError(
            "standalone chk3 release has an unauthorized dataset role"
        )
    if root.get("quality_status") != "passed" or root.get("immutable") is not True:
        raise DatasetReleaseValidationError(
            "standalone chk3 release must be immutable with quality_status=passed"
        )
    if root.get("dag_bindable") is not False:
        raise DatasetReleaseValidationError(
            "standalone chk3 release must explicitly remain non-DAG-bindable"
        )
    blocker = root.get("dag_binding_blocker")
    if not isinstance(blocker, str) or not blocker.strip():
        raise DatasetReleaseValidationError(
            "standalone chk3 release must record its canonical DAG blocker"
        )
    if root.get("training_mapping") != (
        "chk1 final analysis -> reasoning -> formal Minutes paragraph"
    ):
        raise DatasetReleaseValidationError(
            "standalone chk3 training mapping is not the approved Minutes-SFT mapping"
        )
    release_id = root.get("release_id")
    if (
        not isinstance(release_id, str)
        or not release_id
        or release_id != release_root.name
    ):
        raise DatasetReleaseValidationError(
            "standalone chk3 release identity does not match its directory"
        )

    split_counts = _required_mapping(root.get("split_counts"), label="split_counts")
    if set(split_counts) != set(_CHK3_SPLITS):
        raise DatasetReleaseValidationError(
            "standalone chk3 split_counts must contain train/validation/test exactly"
        )
    counts = {
        split: _required_count(split_counts[split], label=f"split_counts.{split}")
        for split in _CHK3_SPLITS
    }
    if any(count <= 0 for count in counts.values()):
        raise DatasetReleaseValidationError(
            "standalone chk3 release requires a non-empty train/validation/test population"
        )
    total_rows = sum(counts.values())
    if _required_count(root.get("total_rows"), label="total_rows") != total_rows:
        raise DatasetReleaseValidationError(
            "standalone chk3 total_rows disagrees with split_counts"
        )

    expected_files = {
        "audits/data_quality.json",
        "chk3_minutes_sft.template.yaml",
        *(f"minutes_alignment/{split}.jsonl" for split in _CHK3_SPLITS),
        *(f"minutes_alignment/manifests/{split}.jsonl" for split in _CHK3_SPLITS),
    }
    files = _required_mapping(root.get("files"), label="files")
    if set(files) != expected_files:
        raise DatasetReleaseValidationError(
            "standalone chk3 release files do not match the v1 sealed file set"
        )
    resolved_files: dict[str, Path] = {}
    for relative_path in sorted(expected_files):
        descriptor = _required_mapping(
            files[relative_path], label=f"files.{relative_path}"
        )
        if descriptor.get("path") != relative_path:
            raise DatasetReleaseValidationError(
                f"files.{relative_path}.path does not match its manifest key"
            )
        file_path = _resolve_release_member(
            release_root,
            descriptor.get("path"),
            label=f"files.{relative_path}.path",
        )
        if file_path.stat().st_size != _required_count(
            descriptor.get("bytes"), label=f"files.{relative_path}.bytes"
        ):
            raise DatasetReleaseValidationError(
                f"standalone chk3 file byte-count mismatch: {relative_path}"
            )
        if sha256_file(file_path) != _required_sha256(
            descriptor.get("sha256"), label=f"files.{relative_path}.sha256"
        ):
            raise DatasetReleaseValidationError(
                f"standalone chk3 file SHA-256 mismatch: {relative_path}"
            )
        resolved_files[relative_path] = file_path

    sample_ids: set[str] = set()
    split_files: dict[str, dict[str, Any]] = {}
    source_split_names = {"train": "train", "validation": "eval", "test": "test"}
    for split in _CHK3_SPLITS:
        data_relative = f"minutes_alignment/{split}.jsonl"
        rows_relative = f"minutes_alignment/manifests/{split}.jsonl"
        data_descriptor = _required_mapping(
            files[data_relative], label=f"files.{data_relative}"
        )
        row_descriptor = _required_mapping(
            files[rows_relative], label=f"files.{rows_relative}"
        )
        expected_rows = counts[split]
        if (
            _required_count(data_descriptor.get("rows"), label=f"{data_relative}.rows")
            != expected_rows
            or _required_count(
                row_descriptor.get("rows"), label=f"{rows_relative}.rows"
            )
            != expected_rows
        ):
            raise DatasetReleaseValidationError(
                f"standalone chk3 {split} row descriptor mismatch"
            )
        data_rows = _read_jsonl_objects(
            resolved_files[data_relative], label=f"standalone chk3 {split} split"
        )
        manifest_rows = _read_jsonl_objects(
            resolved_files[rows_relative],
            label=f"standalone chk3 {split} row manifest",
        )
        if len(data_rows) != expected_rows or len(manifest_rows) != expected_rows:
            raise DatasetReleaseValidationError(
                f"standalone chk3 {split} physical row-count mismatch"
            )
        for index, (data_row, row) in enumerate(zip(data_rows, manifest_rows)):
            row_number = index + 1
            if set(data_row) != {"prompt", "response"}:
                raise DatasetReleaseValidationError(
                    f"standalone chk3 {split} row {row_number} has invalid data fields"
                )
            prompt = data_row.get("prompt")
            response = data_row.get("response")
            if (
                not isinstance(prompt, str)
                or not prompt.strip()
                or not isinstance(response, str)
                or not response.strip()
                or response.count("\n</think>\n") != 1
                or "<think>" in response
            ):
                raise DatasetReleaseValidationError(
                    f"standalone chk3 {split} row {row_number} violates the response boundary contract"
                )
            reasoning, minutes = response.split("\n</think>\n", 1)
            if not reasoning.strip() or not minutes.strip() or "\n" in minutes:
                raise DatasetReleaseValidationError(
                    f"standalone chk3 {split} row {row_number} has an invalid Minutes target"
                )
            sample_id = row.get("sample_id")
            if (
                row.get("schema_version") != STANDALONE_CHK3_RELEASE_SCHEMA
                or row.get("split") != split
                or row.get("source_split") != source_split_names[split]
                or row.get("source_index") != index
                or not isinstance(sample_id, str)
                or not sample_id
                or sample_id in sample_ids
            ):
                raise DatasetReleaseValidationError(
                    f"standalone chk3 {split} row {row_number} has an invalid manifest identity"
                )
            sample_ids.add(sample_id)
            expected_hashes = {
                "prompt_sha256": _sha256_text(prompt),
                "response_sha256": _sha256_text(response),
                "reasoning_sha256": _sha256_text(reasoning),
                "minutes_sha256": _sha256_text(minutes),
            }
            for key, observed in expected_hashes.items():
                if (
                    _required_sha256(
                        row.get(key), label=f"{split} row {row_number}.{key}"
                    )
                    != observed
                ):
                    raise DatasetReleaseValidationError(
                        f"standalone chk3 {split} row {row_number} {key} mismatch"
                    )
            for key in (
                "analysis_sha256",
                "source_analysis_sha256",
                "source_response_sha256",
            ):
                _required_sha256(row.get(key), label=f"{split} row {row_number}.{key}")
            for key in (
                "prompt_tokens",
                "reasoning_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                _required_count(row.get(key), label=f"{split} row {row_number}.{key}")
        split_files[split] = {
            "path": data_relative,
            "rows": expected_rows,
            "sha256": data_descriptor["sha256"],
        }
    if len(sample_ids) != total_rows:
        raise DatasetReleaseValidationError(
            "standalone chk3 sample-ID population is incomplete"
        )

    audit = _read_json_object(
        resolved_files["audits/data_quality.json"],
        label="standalone chk3 data-quality audit",
    )
    if (
        audit.get("schema_version") != STANDALONE_CHK3_RELEASE_SCHEMA
        or audit.get("status") != "passed"
        or audit.get("intended_use")
        != "chk3 SFT: analysis to formal FOMC Minutes paragraph"
        or audit.get("split_counts") != counts
        or audit.get("total_rows") != total_rows
        or audit.get("unique_sample_ids") != total_rows
    ):
        raise DatasetReleaseValidationError(
            "standalone chk3 data-quality audit does not bind the release population"
        )
    for key in (
        "duplicate_sample_ids",
        "missing_required_fields",
        "invalid_response_boundaries",
        "analysis_evidence_citations",
        "reasoning_meta_contamination",
        "transport_wrapped_analyses",
    ):
        if audit.get(key) != 0:
            raise DatasetReleaseValidationError(
                f"standalone chk3 data-quality gate did not pass: {key}"
            )
    checks = _required_mapping(audit.get("checks"), label="data-quality checks")
    if not checks or not all(value == "passed" for value in checks.values()):
        raise DatasetReleaseValidationError(
            "standalone chk3 data-quality audit contains a failed check"
        )
    token_contract = _required_mapping(
        audit.get("token_contract"), label="data-quality token_contract"
    )
    if (
        token_contract.get("total_max") != 4096
        or token_contract.get("truncation") is not False
        or token_contract.get("overflow_policy") != "error"
    ):
        raise DatasetReleaseValidationError(
            "standalone chk3 release does not satisfy the full-completion token contract"
        )
    token_stats = _required_mapping(audit.get("token_stats"), label="token_stats")
    total_stats = _required_mapping(token_stats.get("total"), label="token_stats.total")
    max_total = _required_count(total_stats.get("max"), label="token_stats.total.max")
    if max_total <= 0 or max_total > 4096:
        raise DatasetReleaseValidationError(
            "standalone chk3 audited maximum token count is invalid"
        )

    source = _required_mapping(root.get("source"), label="source")
    for key in (
        "chk1_handoff_sha256",
        "chk3_prompt_contract_sha256",
        "chk3_summary_sha256",
    ):
        _required_sha256(source.get(key), label=f"source.{key}")
    prior = _required_mapping(
        source.get("prior_chk3_release"), label="source.prior_chk3_release"
    )
    for key in ("handoff_sha256", "release_manifest_sha256"):
        _required_sha256(prior.get(key), label=f"source.prior_chk3_release.{key}")

    result = dict(root)
    result["split_files"] = split_files
    result["verified_scope"] = {
        "schema_version": STANDALONE_CHK3_BINDING_SCHEMA,
        "scope_id": STANDALONE_CHK3_DIRECT_SCOPE,
        "training_stage": "chk3",
        "operation": "direct_sft_training",
        "parent_stage": "chk1",
        "canonical_dag_bindable": False,
        "promotable_as_canonical_chk3": False,
        "downstream_stages_allowed": [],
    }
    return result


def _paper_chk2_descriptor_files(
    *, release_root: Path, artifacts: Mapping[str, Any]
) -> dict[str, tuple[Path, Mapping[str, Any]]]:
    """Resolve and authenticate every artifact descriptor in a paper-chk2 release."""

    resolved: dict[str, tuple[Path, Mapping[str, Any]]] = {}

    def visit(value: Any, *, label: str) -> None:
        node = _required_mapping(value, label=label)
        if "path" in node:
            expected_keys = {"path", "bytes", "sha256"}
            path_value = node.get("path")
            if isinstance(path_value, str) and path_value.endswith(".jsonl"):
                expected_keys.add("rows")
            if set(node) != expected_keys:
                raise DatasetReleaseValidationError(
                    f"{label} has an invalid artifact descriptor schema"
                )
            path = _resolve_release_member(
                release_root, path_value, label=f"{label}.path"
            )
            relative = path.relative_to(release_root).as_posix()
            if relative in resolved:
                raise DatasetReleaseValidationError(
                    f"paper chk2 artifact is described more than once: {relative}"
                )
            expected_bytes = _required_count(node.get("bytes"), label=f"{label}.bytes")
            if path.stat().st_size != expected_bytes:
                raise DatasetReleaseValidationError(
                    f"paper chk2 artifact byte-size drift: {relative}"
                )
            expected_sha = _required_sha256(node.get("sha256"), label=f"{label}.sha256")
            if sha256_file(path) != expected_sha:
                raise DatasetReleaseValidationError(
                    f"paper chk2 artifact SHA-256 drift: {relative}"
                )
            if relative.endswith(".jsonl"):
                rows = _read_jsonl_objects(
                    path, label=f"paper chk2 artifact {relative}"
                )
                if len(rows) != _required_count(
                    node.get("rows"), label=f"{label}.rows"
                ):
                    raise DatasetReleaseValidationError(
                        f"paper chk2 artifact row-count drift: {relative}"
                    )
            resolved[relative] = (path, node)
            return
        if not node:
            raise DatasetReleaseValidationError(f"{label} must not be empty")
        for key, nested in node.items():
            if not isinstance(key, str) or not key:
                raise DatasetReleaseValidationError(
                    f"{label} contains an invalid artifact key"
                )
            visit(nested, label=f"{label}.{key}")

    visit(artifacts, label="artifacts")

    expected_files = {
        path.relative_to(release_root).as_posix()
        for path in release_root.rglob("*")
        if path.is_file()
    }
    for path in release_root.rglob("*"):
        if path.is_symlink():
            raise DatasetReleaseValidationError(
                f"paper chk2 release contains a symlink: {path}"
            )
    expected_files -= {"release_manifest.json", "handoff.json"}
    if set(resolved) != expected_files:
        missing = sorted(expected_files - set(resolved))
        orphan_descriptors = sorted(set(resolved) - expected_files)
        raise DatasetReleaseValidationError(
            "paper chk2 artifact inventory drift; "
            f"undescribed={missing}, orphan_descriptors={orphan_descriptors}"
        )
    return resolved


def _paper_chk2_extract_analysis(prompt: str, *, label: str) -> str:
    if not prompt.startswith(PAPER_CHK2_USER_PROMPT_PREFIX):
        raise DatasetReleaseValidationError(
            f"{label} does not use the approved student user-prompt prefix"
        )
    serialized = prompt[len(PAPER_CHK2_USER_PROMPT_PREFIX) :]
    try:
        payload = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise DatasetReleaseValidationError(
            f"{label} has an invalid user-prompt JSON boundary"
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != {"analysis"}:
        raise DatasetReleaseValidationError(
            f"{label} user prompt must contain only analysis"
        )
    analysis = payload.get("analysis")
    if (
        not isinstance(analysis, str)
        or not analysis.strip()
        or analysis != analysis.strip()
    ):
        raise DatasetReleaseValidationError(f"{label} source analysis is invalid")
    if serialized != _canonical_json({"analysis": analysis}):
        raise DatasetReleaseValidationError(
            f"{label} user-prompt JSON is not in canonical form"
        )
    return analysis


def _verify_paper_chk2_parent_binding(
    *,
    release_root: Path,
    root: Mapping[str, Any],
    artifact_files: Mapping[str, tuple[Path, Mapping[str, Any]]],
    model_path: str | Path,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    expected_model = (repo_root / PAPER_CHK2_PARENT_MODEL_RELATIVE).resolve(strict=True)
    model = _canonical_path(
        model_path, label="paper chk2 model_name_or_path", directory=True
    )
    if model != expected_model:
        raise DatasetReleaseValidationError(
            "paper chk2 must use the exact authorized chk1 checkpoint-200 merge"
        )
    try:
        fingerprint = fingerprint_artifact_path(model)
    except (OSError, ValueError) as exc:
        raise DatasetReleaseValidationError(
            "cannot fingerprint the paper chk2 parent model"
        ) from exc
    if (
        fingerprint.get("kind") != "directory"
        or fingerprint.get("sha256") != PAPER_CHK2_PARENT_MODEL_SHA256
    ):
        raise DatasetReleaseValidationError(
            "paper chk2 parent model directory digest drift"
        )

    parent = _required_mapping(root.get("parent_checkpoint"), label="parent_checkpoint")
    expected_parent_values = {
        "schema_version": PAPER_CHK2_PARENT_MANIFEST_SCHEMA,
        "model_path": PAPER_CHK2_PARENT_MODEL_RELATIVE,
        "model_sha256": PAPER_CHK2_PARENT_MODEL_SHA256,
        "authorization_schema_version": PAPER_CHK2_PARENT_AUTHORIZATION_SCHEMA,
        "authorization_sha256": PAPER_CHK2_PARENT_AUTHORIZATION_SHA256,
        "allowed_stage": "chk2",
        "further_downstream_stages_allowed": [],
    }
    for key, expected in expected_parent_values.items():
        if parent.get(key) != expected:
            raise DatasetReleaseValidationError(
                f"paper chk2 parent checkpoint binding drift: {key}"
            )

    checkpoint_path = artifact_files.get("provenance/parent_checkpoint_manifest.json")
    authorization_path = artifact_files.get("provenance/parent_authorization.json")
    if checkpoint_path is None or authorization_path is None:
        raise DatasetReleaseValidationError(
            "paper chk2 release is missing parent provenance"
        )
    checkpoint_file = checkpoint_path[0]
    authorization_file = authorization_path[0]
    if (
        parent.get("checkpoint_manifest_file_sha256") != sha256_file(checkpoint_file)
        or sha256_file(checkpoint_file) != PAPER_CHK2_PARENT_MANIFEST_FILE_SHA256
    ):
        raise DatasetReleaseValidationError(
            "paper chk2 parent checkpoint manifest hash drift"
        )
    checkpoint_manifest = _read_json_object(
        checkpoint_file, label="paper chk2 parent checkpoint manifest"
    )
    unsigned_checkpoint = dict(checkpoint_manifest)
    checkpoint_integrity = unsigned_checkpoint.pop("integrity", None)
    model_fingerprint = _required_mapping(
        checkpoint_manifest.get("model_fingerprint"),
        label="paper chk2 parent checkpoint model_fingerprint",
    )
    checkpoint_scope = _required_mapping(
        checkpoint_manifest.get("scope"), label="paper chk2 parent checkpoint scope"
    )
    checkpoint_authorization = _required_mapping(
        checkpoint_manifest.get("authorization"),
        label="paper chk2 parent checkpoint authorization",
    )
    if (
        checkpoint_manifest.get("schema_version") != PAPER_CHK2_PARENT_MANIFEST_SCHEMA
        or checkpoint_manifest.get("status")
        != "ready_for_chk2_parent_under_explicit_override"
        or not isinstance(checkpoint_integrity, Mapping)
        or checkpoint_integrity.get("payload_sha256")
        != _sha256_text(_canonical_json(unsigned_checkpoint))
        or model_fingerprint.get("sha256") != PAPER_CHK2_PARENT_MODEL_SHA256
        or model_fingerprint.get("file_count") != fingerprint.get("file_count")
        or model_fingerprint.get("total_bytes") != fingerprint.get("total_bytes")
        or checkpoint_scope.get("allowed_stage") != "chk2"
        or checkpoint_scope.get("further_downstream_stages_allowed") != []
        or checkpoint_authorization.get("authorization_sha256")
        != PAPER_CHK2_PARENT_AUTHORIZATION_SHA256
        or checkpoint_authorization.get("file_sha256")
        != PAPER_CHK2_PARENT_AUTHORIZATION_FILE_SHA256
    ):
        raise DatasetReleaseValidationError(
            "paper chk2 parent checkpoint manifest content drift"
        )
    if parent.get("authorization_file_sha256") != sha256_file(authorization_file):
        raise DatasetReleaseValidationError(
            "paper chk2 copied parent authorization hash drift"
        )
    if sha256_file(authorization_file) != PAPER_CHK2_PARENT_AUTHORIZATION_FILE_SHA256:
        raise DatasetReleaseValidationError(
            "paper chk2 parent authorization is not the pinned receipt"
        )
    authorization = _read_json_object(
        authorization_file, label="paper chk2 parent authorization"
    )
    unsigned_authorization = dict(authorization)
    stored_authorization_sha = unsigned_authorization.pop("authorization_sha256", None)
    if (
        stored_authorization_sha != PAPER_CHK2_PARENT_AUTHORIZATION_SHA256
        or stored_authorization_sha
        != _sha256_text(_canonical_json(unsigned_authorization))
        or authorization.get("schema_version") != PAPER_CHK2_PARENT_AUTHORIZATION_SCHEMA
        or authorization.get("status") != "authorized"
    ):
        raise DatasetReleaseValidationError(
            "paper chk2 parent authorization content drift"
        )
    scope = _required_mapping(
        authorization.get("scope"), label="paper chk2 parent authorization scope"
    )
    if (
        scope.get("source_stage") != "chk1"
        or scope.get("target_stage") != "chk2"
        or scope.get("downstream_stages_allowed") != ["chk2"]
        or scope.get("further_downstream_stages_allowed") != []
        or "use_merged_checkpoint_200_as_chk2_parent"
        not in scope.get("allowed_operations", [])
    ):
        raise DatasetReleaseValidationError(
            "paper chk2 parent authorization does not authorize chk2 parent use"
        )
    bindings = _required_mapping(
        authorization.get("bindings"), label="paper chk2 parent authorization bindings"
    )
    if bindings.get("merged_destination") != PAPER_CHK2_PARENT_MODEL_RELATIVE:
        raise DatasetReleaseValidationError(
            "paper chk2 parent authorization destination drift"
        )

    canonical_authorization = _canonical_path(
        repo_root / PAPER_CHK2_PARENT_AUTHORIZATION_RELATIVE,
        label="canonical paper chk2 parent authorization",
        directory=False,
    )
    if (
        sha256_file(canonical_authorization)
        != PAPER_CHK2_PARENT_AUTHORIZATION_FILE_SHA256
        or canonical_authorization.read_bytes() != authorization_file.read_bytes()
    ):
        raise DatasetReleaseValidationError(
            "paper chk2 release authorization differs from the canonical receipt"
        )
    canonical_checkpoint = _canonical_path(
        repo_root / PAPER_CHK2_PARENT_MANIFEST_RELATIVE,
        label="canonical paper chk2 parent checkpoint manifest",
        directory=False,
    )
    if (
        sha256_file(canonical_checkpoint) != PAPER_CHK2_PARENT_MANIFEST_FILE_SHA256
        or canonical_checkpoint.read_bytes() != checkpoint_file.read_bytes()
    ):
        raise DatasetReleaseValidationError(
            "paper chk2 release checkpoint manifest differs from the canonical receipt"
        )
    return {
        "model": fingerprint,
        "authorization": {
            "path": str(canonical_authorization),
            "file_sha256": PAPER_CHK2_PARENT_AUTHORIZATION_FILE_SHA256,
            "authorization_sha256": PAPER_CHK2_PARENT_AUTHORIZATION_SHA256,
            "allowed_stage": "chk2",
        },
    }


def verify_paper_chk2_sft_release(
    *,
    dataset_dir: str | Path,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    training_scope: str,
    system_prompt: str | None,
    model_path: str | Path,
) -> dict[str, Any]:
    """Bind the recovery-v1 paper chk2 Minutes release to direct SFT.

    The test split and all official-reference artifacts are authenticated, but
    only the strict ``{prompt,response}`` train and validation files are
    returned to the training loader.
    """

    if training_scope != PAPER_CHK2_TRAINING_SCOPE:
        raise DatasetReleaseValidationError(
            "paper chk2 training scope is not the approved non-DAG Minutes-SFT scope"
        )
    expected_sha = _required_sha256(
        expected_manifest_sha256,
        label="dataset_paper_chk2_release_manifest_sha256",
    )
    dataset = _canonical_path(dataset_dir, label="dataset_name", directory=True)
    manifest = _canonical_path(
        manifest_path,
        label="dataset_paper_chk2_release_manifest",
        directory=False,
    )
    release_root = manifest.parent
    if dataset != release_root / "minutes_alignment":
        raise DatasetReleaseValidationError(
            "dataset_name must be the minutes_alignment directory beside the "
            "paper chk2 release manifest"
        )
    if sha256_file(manifest) != expected_sha:
        raise DatasetReleaseValidationError(
            "paper chk2 release manifest SHA-256 disagrees with the training config"
        )
    if (
        system_prompt != PAPER_CHK2_STUDENT_SYSTEM_PROMPT
        or _sha256_text(system_prompt or "") != PAPER_CHK2_STUDENT_SYSTEM_PROMPT_SHA256
    ):
        raise DatasetReleaseValidationError(
            "paper chk2 system_prompt is not the sealed permissive-reasoning prompt"
        )
    if (
        _sha256_text(PAPER_CHK2_USER_PROMPT_TEMPLATE)
        != PAPER_CHK2_USER_PROMPT_TEMPLATE_SHA256
    ):
        raise DatasetReleaseValidationError(
            "paper chk2 compiled user-prompt template digest drift"
        )

    root = _read_json_object(manifest, label="paper chk2 release manifest")
    unsigned_manifest = dict(root)
    stored_manifest_sha = unsigned_manifest.pop("manifest_sha256", None)
    if (
        root.get("schema_version") != PAPER_CHK2_RELEASE_SCHEMA
        or root.get("status") != "complete"
        or stored_manifest_sha != _sha256_text(_canonical_json(unsigned_manifest))
    ):
        raise DatasetReleaseValidationError(
            "paper chk2 release manifest schema/status/self-hash drift"
        )
    exact_manifest_values = {
        "quality_status": "passed",
        "dataset_role": PAPER_CHK2_DATASET_ROLE,
        "training_scope": PAPER_CHK2_TRAINING_SCOPE,
        "immutable": True,
        "training_ready": True,
        "training_only": True,
        "evaluation_eligible": False,
        "dag_bindable": False,
        "promotable_as_canonical_chk2": False,
        "source_rows": 1743,
        "split_pass_counts": {"train": 305, "validation": 42, "test": 44},
    }
    for key, expected in exact_manifest_values.items():
        if root.get(key) != expected:
            raise DatasetReleaseValidationError(
                f"paper chk2 release manifest policy drift: {key}"
            )

    prompt_summary = _required_mapping(
        root.get("student_prompt_contract"), label="student_prompt_contract"
    )
    if dict(prompt_summary) != {
        "system_prompt_sha256": PAPER_CHK2_STUDENT_SYSTEM_PROMPT_SHA256,
        "user_prompt_template_sha256": PAPER_CHK2_USER_PROMPT_TEMPLATE_SHA256,
        "response_boundary": "</think>",
    }:
        raise DatasetReleaseValidationError(
            "paper chk2 manifest student-prompt contract drift"
        )

    artifacts = _required_mapping(root.get("artifacts"), label="artifacts")
    if set(artifacts) != {"minutes_alignment", "audits", "provenance"}:
        raise DatasetReleaseValidationError("paper chk2 artifact groups drift")
    minutes_artifacts = _required_mapping(
        artifacts.get("minutes_alignment"), label="artifacts.minutes_alignment"
    )
    if set(minutes_artifacts) != set(_CHK3_SPLITS):
        raise DatasetReleaseValidationError("paper chk2 split artifact groups drift")
    audit_artifacts = _required_mapping(
        artifacts.get("audits"), label="artifacts.audits"
    )
    expected_audits = {
        "rejections",
        "source_admission",
        "validator_a",
        "validator_b",
        "repair_history",
        "compatibility_replay",
        "evidence_ledger",
        "tokenizer_replay",
        "data_quality",
    }
    if set(audit_artifacts) != expected_audits:
        raise DatasetReleaseValidationError("paper chk2 audit artifact groups drift")
    provenance_artifacts = _required_mapping(
        artifacts.get("provenance"), label="artifacts.provenance"
    )
    expected_provenance = {
        "student_prompt_contract",
        "source_handoff_manifest",
        "source_admission_receipt",
        "source_prompt_contract",
        "official_pre_action_reference_bank",
        "recovery_partial_handoff_manifest",
        "recovery_receipt",
        "recovery_prompt_contract",
        "recovery_attempts",
        "parent_checkpoint_manifest",
        "parent_authorization",
    }
    if set(provenance_artifacts) != expected_provenance:
        raise DatasetReleaseValidationError(
            "paper chk2 provenance artifact groups drift"
        )
    recovery_attempts = _required_mapping(
        provenance_artifacts.get("recovery_attempts"),
        label="artifacts.provenance.recovery_attempts",
    )
    if not recovery_attempts:
        raise DatasetReleaseValidationError(
            "paper chk2 release must bind at least one recovery attempt"
        )
    artifact_files = _paper_chk2_descriptor_files(
        release_root=release_root, artifacts=artifacts
    )

    handoff_path = _canonical_path(
        release_root / "handoff.json", label="paper chk2 handoff", directory=False
    )
    handoff = _read_json_object(handoff_path, label="paper chk2 handoff")
    unsigned_handoff = dict(handoff)
    stored_handoff_sha = unsigned_handoff.pop("handoff_sha256", None)
    if (
        handoff.get("schema_version") != PAPER_CHK2_HANDOFF_SCHEMA
        or handoff.get("status") != "complete"
        or stored_handoff_sha != _sha256_text(_canonical_json(unsigned_handoff))
        or handoff.get("release_manifest_sha256") != stored_manifest_sha
        or handoff.get("release_manifest_file_sha256") != expected_sha
        or handoff.get("parent_checkpoint_model_sha256")
        != PAPER_CHK2_PARENT_MODEL_SHA256
        or handoff.get("parent_authorization_sha256")
        != PAPER_CHK2_PARENT_AUTHORIZATION_SHA256
        or handoff.get("parent_authorization_file_sha256")
        != PAPER_CHK2_PARENT_AUTHORIZATION_FILE_SHA256
    ):
        raise DatasetReleaseValidationError("paper chk2 release handoff drift")

    prompt_contract_record = artifact_files.get("prompt_contract.json")
    if prompt_contract_record is None:
        raise DatasetReleaseValidationError(
            "paper chk2 release is missing its student prompt contract"
        )
    prompt_contract = _read_json_object(
        prompt_contract_record[0], label="paper chk2 student prompt contract"
    )
    if dict(prompt_contract) != {
        "schema_version": PAPER_CHK2_PROMPT_CONTRACT_SCHEMA,
        "system_prompt": PAPER_CHK2_STUDENT_SYSTEM_PROMPT,
        "system_prompt_sha256": PAPER_CHK2_STUDENT_SYSTEM_PROMPT_SHA256,
        "user_prompt_template": PAPER_CHK2_USER_PROMPT_TEMPLATE,
        "user_prompt_template_sha256": PAPER_CHK2_USER_PROMPT_TEMPLATE_SHA256,
        "response_boundary": "</think>",
        "opening_think_supplied_by_chat_template": True,
    }:
        raise DatasetReleaseValidationError(
            "paper chk2 student prompt contract content drift"
        )

    parent_binding = _verify_paper_chk2_parent_binding(
        release_root=release_root,
        root=root,
        artifact_files=artifact_files,
        model_path=model_path,
    )

    split_counts = dict(root["split_pass_counts"])
    sample_ids: set[str] = set()
    source_indexes: set[tuple[str, int]] = set()
    meeting_dates: dict[str, set[str]] = {split: set() for split in _CHK3_SPLITS}
    pass_order: list[tuple[str, str]] = []
    split_files: dict[str, Path] = {}
    sealed_test: dict[str, Any] | None = None
    pass_status_counts: dict[str, int] = {}
    sidecar_ids_by_split: dict[str, list[str]] = {}
    for split in _CHK3_SPLITS:
        group = _required_mapping(
            minutes_artifacts.get(split),
            label=f"artifacts.minutes_alignment.{split}",
        )
        if set(group) != {"data", "manifest"}:
            raise DatasetReleaseValidationError(
                f"paper chk2 {split} split descriptor group drift"
            )
        expected_data_path = f"minutes_alignment/{split}.jsonl"
        expected_sidecar_path = f"minutes_alignment/manifests/{split}.jsonl"
        data_descriptor = _required_mapping(
            group.get("data"), label=f"paper chk2 {split} data descriptor"
        )
        sidecar_descriptor = _required_mapping(
            group.get("manifest"), label=f"paper chk2 {split} sidecar descriptor"
        )
        if (
            data_descriptor.get("path") != expected_data_path
            or sidecar_descriptor.get("path") != expected_sidecar_path
            or data_descriptor.get("rows") != split_counts[split]
            or sidecar_descriptor.get("rows") != split_counts[split]
        ):
            raise DatasetReleaseValidationError(
                f"paper chk2 {split} split descriptor drift"
            )
        data_path = artifact_files[expected_data_path][0]
        sidecar_path = artifact_files[expected_sidecar_path][0]
        data_rows = _read_jsonl_objects(data_path, label=f"paper chk2 {split} data")
        sidecar_rows = _read_jsonl_objects(
            sidecar_path, label=f"paper chk2 {split} sidecars"
        )
        if len(data_rows) != split_counts[split] or len(sidecar_rows) != len(data_rows):
            raise DatasetReleaseValidationError(
                f"paper chk2 {split} physical row-count drift"
            )
        sidecar_ids_by_split[split] = []
        for index, (data_row, sidecar) in enumerate(zip(data_rows, sidecar_rows)):
            label = f"paper chk2 {split} row {index + 1}"
            if set(data_row) != {"prompt", "response"}:
                raise DatasetReleaseValidationError(
                    f"{label} training data must contain prompt/response only"
                )
            expected_sidecar_keys = {
                "sample_id",
                "split",
                "source_index",
                "meeting_date",
                "atomic_topic",
                "section_style_id",
                "terminal_status",
                "source_analysis_sha256",
                "prompt_sha256",
                "response_sha256",
                "lineage",
                "release_index",
            }
            if set(sidecar) != expected_sidecar_keys:
                raise DatasetReleaseValidationError(f"{label} sidecar schema drift")
            prompt = data_row.get("prompt")
            response = data_row.get("response")
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise DatasetReleaseValidationError(f"{label} contains non-text data")
            analysis = _paper_chk2_extract_analysis(prompt, label=label)
            if (
                response.count("\n</think>\n") != 1
                or response.count("</think>") != 1
                or "<think>" in response
            ):
                raise DatasetReleaseValidationError(
                    f"{label} violates the single reasoning-boundary contract"
                )
            reasoning, minutes = response.split("\n</think>\n", 1)
            if (
                not reasoning.strip()
                or not minutes.strip()
                or minutes != minutes.strip()
                or "\n" in minutes
                or not 20 <= len(PAPER_CHK2_WORD_RE.findall(minutes)) <= 400
            ):
                raise DatasetReleaseValidationError(
                    f"{label} has an invalid single-paragraph Minutes target"
                )
            sample_id = sidecar.get("sample_id")
            source_index = sidecar.get("source_index")
            meeting_date = sidecar.get("meeting_date")
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or sample_id in sample_ids
                or isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or source_index < 0
                or (split, source_index) in source_indexes
                or not isinstance(meeting_date, str)
                or not meeting_date
                or sidecar.get("split") != split
                or sidecar.get("release_index") != index
                or sidecar.get("terminal_status") != "PASS"
            ):
                raise DatasetReleaseValidationError(f"{label} identity/order drift")
            for key in ("atomic_topic", "section_style_id"):
                if not isinstance(sidecar.get(key), str) or not sidecar[key]:
                    raise DatasetReleaseValidationError(f"{label} has an invalid {key}")
            expected_hashes = {
                "source_analysis_sha256": _sha256_text(analysis),
                "prompt_sha256": _sha256_text(prompt),
                "response_sha256": _sha256_text(response),
            }
            for key, expected in expected_hashes.items():
                if sidecar.get(key) != expected:
                    raise DatasetReleaseValidationError(f"{label} {key} mismatch")
            lineage = _required_mapping(
                sidecar.get("lineage"), label=f"{label}.lineage"
            )
            required_lineage = {
                "source_analysis_is_exact_chk1_final_answer": True,
                "source_analysis_was_repaired": False,
                "c8_used_for_training": False,
                "rewrite_teacher_saw_official_minutes": False,
                "target_is_teacher_synthetic_rewrite": True,
                "validator_a_is_factual_gate": True,
                "validator_b_is_style_gate": True,
                "validator_b_used_for_training_selection": True,
                "official_minutes_used_as_student_target": False,
                "training_only": True,
                "evaluation_eligible": False,
                "suitable_for_leakage_safe_evaluation": False,
            }
            for key, expected in required_lineage.items():
                if lineage.get(key) is not expected:
                    raise DatasetReleaseValidationError(f"{label} lineage drift: {key}")
            sample_ids.add(sample_id)
            source_indexes.add((split, source_index))
            meeting_dates[split].add(meeting_date)
            sidecar_ids_by_split[split].append(sample_id)
            pass_order.append((sample_id, split))
        pass_status_counts[split] = len(data_rows)
        if split == "test":
            sealed_test = {
                "path": str(data_path),
                "rows": len(data_rows),
                "sha256": data_descriptor["sha256"],
            }
        else:
            split_files[split] = data_path

    if len(sample_ids) != 391 or len(pass_order) != 391:
        raise DatasetReleaseValidationError(
            "paper chk2 PASS population is not exactly 391 unique samples"
        )
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if meeting_dates[left] & meeting_dates[right]:
            raise DatasetReleaseValidationError(
                f"paper chk2 meeting split leakage between {left} and {right}"
            )

    token_path = artifact_files["audits/tokenizer_replay.jsonl"][0]
    token_rows = _read_jsonl_objects(token_path, label="paper chk2 tokenizer replay")
    expected_token_keys = {
        "sample_id",
        "split",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "single_bos",
        "single_eos",
        "no_truncation",
        "completion_only_prompt_masked",
        "completion_mask_covers_reasoning_boundary_answer_eos",
    }
    if len(token_rows) != 391:
        raise DatasetReleaseValidationError(
            "paper chk2 tokenizer replay population drift"
        )
    max_total_tokens = 0
    for index, (token_row, identity) in enumerate(zip(token_rows, pass_order)):
        if set(token_row) != expected_token_keys:
            raise DatasetReleaseValidationError(
                f"paper chk2 tokenizer replay row {index + 1} schema drift"
            )
        if (token_row.get("sample_id"), token_row.get("split")) != identity:
            raise DatasetReleaseValidationError(
                f"paper chk2 tokenizer replay row {index + 1} order drift"
            )
        prompt_tokens = _required_count(
            token_row.get("prompt_tokens"), label=f"token replay {index}.prompt_tokens"
        )
        completion_tokens = _required_count(
            token_row.get("completion_tokens"),
            label=f"token replay {index}.completion_tokens",
        )
        total_tokens = _required_count(
            token_row.get("total_tokens"), label=f"token replay {index}.total_tokens"
        )
        if (
            prompt_tokens <= 0
            or completion_tokens <= 0
            or total_tokens != prompt_tokens + completion_tokens
            or total_tokens > 4096
            or any(
                token_row.get(key) is not True
                for key in (
                    "single_bos",
                    "single_eos",
                    "no_truncation",
                    "completion_only_prompt_masked",
                    "completion_mask_covers_reasoning_boundary_answer_eos",
                )
            )
        ):
            raise DatasetReleaseValidationError(
                f"paper chk2 tokenizer replay row {index + 1} failed its token gate"
            )
        max_total_tokens = max(max_total_tokens, total_tokens)
    if max_total_tokens != 3316:
        raise DatasetReleaseValidationError(
            "paper chk2 tokenizer replay maximum is not the sealed 3,316 tokens"
        )

    audit_rows = {
        name: _read_jsonl_objects(
            artifact_files[f"audits/{name}.jsonl"][0], label=f"paper chk2 {name} audit"
        )
        for name in (
            "rejections",
            "source_admission",
            "validator_a",
            "validator_b",
            "repair_history",
            "evidence_ledger",
        )
    }
    ledger_ids = [
        str(row.get("sample_id", "")) for row in audit_rows["evidence_ledger"]
    ]
    reject_ids = [str(row.get("sample_id", "")) for row in audit_rows["rejections"]]
    source_ids = [
        str(row.get("sample_id", "")) for row in audit_rows["source_admission"]
    ]
    if (
        len(ledger_ids) != 1743
        or len(set(ledger_ids)) != 1743
        or len(reject_ids) != 1352
        or len(set(reject_ids)) != 1352
        or set(reject_ids) & sample_ids
        or set(reject_ids) | sample_ids != set(ledger_ids)
        or len(source_ids) != 1743
        or set(source_ids) != set(ledger_ids)
    ):
        raise DatasetReleaseValidationError(
            "paper chk2 PASS/REJECT/evidence-ledger population drift"
        )
    for name in ("validator_a", "validator_b", "repair_history"):
        ids = [str(row.get("sample_id", "")) for row in audit_rows[name]]
        if len(ids) != len(set(ids)) or not set(ids) <= set(ledger_ids):
            raise DatasetReleaseValidationError(
                f"paper chk2 {name} audit population drift"
            )

    quality = _read_json_object(
        artifact_files["audits/data_quality.json"][0],
        label="paper chk2 data-quality audit",
    )
    if (
        quality.get("schema_version") != PAPER_CHK2_RELEASE_SCHEMA
        or quality.get("source_rows") != 1743
        or quality.get("split_pass_counts") != split_counts
        or quality.get("rejected_rows") != 1352
        or quality.get("pass_rows") != 391
        or quality.get("three_pass_splits_nonempty") is not True
        or quality.get("evidence_ledger_rows") != 1743
        or quality.get("compatibility_all_rejected") is not True
        or quality.get("validator_a_rows") != len(audit_rows["validator_a"])
        or quality.get("validator_b_rows") != len(audit_rows["validator_b"])
    ):
        raise DatasetReleaseValidationError(
            "paper chk2 data-quality audit population/policy drift"
        )
    compatibility_count = _required_count(
        quality.get("compatibility_count"), label="data_quality.compatibility_count"
    )
    recovery = _required_mapping(root.get("recovery"), label="recovery")
    compatibility_ids = recovery.get("compatibility_ids")
    if (
        compatibility_count != 22
        or not isinstance(compatibility_ids, list)
        or len(compatibility_ids) != 22
        or compatibility_ids != sorted(compatibility_ids)
        or len(set(compatibility_ids)) != 22
        or not all(isinstance(value, str) and value for value in compatibility_ids)
        or recovery.get("compatibility_id_digest")
        != "e62415eecfdb37df240ab7259aba9e244d8fcdfbc8b3fecf907aaf70c5fd5ca7"
    ):
        raise DatasetReleaseValidationError(
            "paper chk2 compatibility-replay binding drift"
        )
    compatibility_rows = _read_jsonl_objects(
        artifact_files["audits/compatibility_replay.jsonl"][0],
        label="paper chk2 compatibility replay",
    )
    compatibility_keys = {
        "sample_id",
        "split",
        "terminal_status",
        "rejection_stage",
        "terminal_record_sha256",
    }
    if (
        len(compatibility_rows) != 22
        or [row.get("sample_id") for row in compatibility_rows] != compatibility_ids
    ):
        raise DatasetReleaseValidationError(
            "paper chk2 compatibility-replay rows drift"
        )
    for row in compatibility_rows:
        if (
            set(row) != compatibility_keys
            or row.get("sample_id") not in set(reject_ids)
            or row.get("split") not in _CHK3_SPLITS
            or row.get("terminal_status") == "PASS"
            or not isinstance(row.get("rejection_stage"), str)
            or not row.get("rejection_stage")
        ):
            raise DatasetReleaseValidationError(
                "paper chk2 compatibility sample was not preserved as a REJECT"
            )
        _required_sha256(
            row.get("terminal_record_sha256"),
            label=f"compatibility {row.get('sample_id')}.terminal_record_sha256",
        )

    split_data_sha = {
        split: minutes_artifacts[split]["data"]["sha256"] for split in _CHK3_SPLITS
    }
    if handoff.get("split_data_sha256") != split_data_sha:
        raise DatasetReleaseValidationError(
            "paper chk2 handoff split-data binding drift"
        )

    return {
        "schema_version": PAPER_CHK2_BINDING_SCHEMA,
        "release_schema_version": PAPER_CHK2_RELEASE_SCHEMA,
        "dataset_role": PAPER_CHK2_DATASET_ROLE,
        "training_scope": PAPER_CHK2_TRAINING_SCOPE,
        "release_root": str(release_root),
        "release_manifest": {
            "path": str(manifest),
            "file_sha256": expected_sha,
            "manifest_sha256": stored_manifest_sha,
        },
        "split_counts": split_counts,
        "split_files": split_files,
        "sealed_test": sealed_test,
        "test_verified_but_not_loaded": True,
        "student_prompt_contract": dict(prompt_contract),
        "parent_binding": parent_binding,
        "token_audit": {
            "rows": len(token_rows),
            "max_total_tokens": max_total_tokens,
            "max_length": 4096,
            "single_bos": True,
            "single_eos": True,
            "no_truncation": True,
            "completion_only_prompt_masked": True,
            "completion_mask_covers_reasoning_boundary_answer_eos": True,
        },
        "quality_audit": {
            "source_rows": 1743,
            "pass_rows": 391,
            "rejected_rows": 1352,
            "compatibility_count": compatibility_count,
            "compatibility_all_rejected": True,
        },
        "scope": {
            "training_stage": "chk2",
            "operation": "minutes_sft_training",
            "parent_stage": "chk1",
            "canonical_dag_bindable": False,
            "promotable_as_canonical_chk2": False,
            "evaluation_eligible": False,
            "test_is_authenticated_but_not_loaded": True,
        },
    }


def _verify_chk1_override_candidate(
    *,
    dataset: Path,
    manifest: Path,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    expected_sha = _required_sha256(
        expected_manifest_sha256,
        label="dataset_semantic_override_candidate_manifest_sha256",
    )
    if sha256_file(manifest) != expected_sha:
        raise DatasetReleaseValidationError(
            "chk1 override candidate manifest SHA-256 disagrees with the training config"
        )
    root = _read_json_object(manifest, label="chk1 override candidate manifest")
    candidate_root = manifest.parent
    if dataset.parent != candidate_root:
        raise DatasetReleaseValidationError(
            "dataset_name and chk1 override candidate manifest must belong to the same candidate"
        )
    if (
        root.get("schema_version") != CHK1_OVERRIDE_CANDIDATE_SCHEMA
        or root.get("quality_status") != "pending_semantic_audit"
        or root.get("immutable_candidate") is not True
    ):
        raise DatasetReleaseValidationError(
            "chk1 override requires an immutable pending_semantic_audit candidate"
        )
    candidate_id = root.get("candidate_id")
    if (
        not isinstance(candidate_id, str)
        or not candidate_id
        or candidate_id != candidate_root.name
    ):
        raise DatasetReleaseValidationError("chk1 override candidate identity mismatch")

    split_counts = _required_mapping(root.get("split_counts"), label="split_counts")
    split_files = _required_mapping(root.get("split_files"), label="split_files")
    if dict(split_counts) != _CHK1_OVERRIDE_SPLIT_COUNTS or set(split_files) != set(
        _SPLITS
    ):
        raise DatasetReleaseValidationError(
            "chk1 override split counts must be exactly train=1354/eval=199/test=190"
        )
    for split in _SPLITS:
        descriptor = _required_mapping(
            split_files.get(split), label=f"split_files.{split}"
        )
        path = _resolve_release_member(
            candidate_root,
            descriptor.get("path"),
            label=f"split_files.{split}.path",
        )
        if path != dataset / f"{split}.jsonl":
            raise DatasetReleaseValidationError(
                f"chk1 override {split} split does not bind dataset_name"
            )
        expected_rows = _CHK1_OVERRIDE_SPLIT_COUNTS[split]
        if (
            _required_count(descriptor.get("rows"), label=f"split_files.{split}.rows")
            != expected_rows
        ):
            raise DatasetReleaseValidationError(
                f"chk1 override {split} split row descriptor mismatch"
            )
        if sha256_file(path) != _required_sha256(
            descriptor.get("sha256"), label=f"split_files.{split}.sha256"
        ):
            raise DatasetReleaseValidationError(
                f"chk1 override {split} split SHA-256 mismatch"
            )
        if (
            len(_read_jsonl_objects(path, label=f"chk1 override {split} split"))
            != expected_rows
        ):
            raise DatasetReleaseValidationError(
                f"chk1 override {split} split physical row-count mismatch"
            )

    if (
        _required_count(root.get("changed_rows"), label="changed_rows")
        != _CHK1_OVERRIDE_CHANGED_ROWS
    ):
        raise DatasetReleaseValidationError("chk1 override changed-row count mismatch")
    repair = _required_mapping(root.get("repair_manifest"), label="repair_manifest")
    repair_path = _resolve_release_member(
        candidate_root, repair.get("path"), label="repair_manifest.path"
    )
    if sha256_file(repair_path) != _required_sha256(
        repair.get("sha256"), label="repair_manifest.sha256"
    ):
        raise DatasetReleaseValidationError(
            "chk1 override repair manifest SHA-256 mismatch"
        )
    total_rows = sum(_CHK1_OVERRIDE_SPLIT_COUNTS.values())
    if _required_count(repair.get("rows"), label="repair_manifest.rows") != total_rows:
        raise DatasetReleaseValidationError(
            "chk1 override repair manifest row-count mismatch"
        )
    repair_rows = _read_jsonl_objects(
        repair_path, label="chk1 override repair manifest"
    )
    if len(repair_rows) != total_rows:
        raise DatasetReleaseValidationError(
            "chk1 override repair manifest physical row-count mismatch"
        )

    changed_ids: set[str] = set()
    all_ids: set[str] = set()
    split_positions = {split: 0 for split in _SPLITS}
    for number, row in enumerate(repair_rows, 1):
        sample_id = row.get("sample_id")
        split = row.get("split")
        line_number = row.get("source_line_number")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in all_ids
            or split not in _SPLITS
            or isinstance(line_number, bool)
            or not isinstance(line_number, int)
        ):
            raise DatasetReleaseValidationError(
                f"chk1 override repair manifest row {number} has invalid identity"
            )
        split_positions[str(split)] += 1
        if line_number != split_positions[str(split)]:
            raise DatasetReleaseValidationError(
                f"chk1 override repair manifest row {number} is out of split order"
            )
        all_ids.add(sample_id)
        hashes = {
            name: _required_sha256(
                row.get(name), label=f"repair manifest row {number}.{name}"
            )
            for name in (
                "prompt_sha256",
                "provided_data_sha256",
                "old_response_sha256",
                "new_response_sha256",
            )
        }
        if hashes["old_response_sha256"] != hashes["new_response_sha256"]:
            changed_ids.add(sample_id)
    if (
        split_positions != _CHK1_OVERRIDE_SPLIT_COUNTS
        or len(changed_ids) != _CHK1_OVERRIDE_CHANGED_ROWS
    ):
        raise DatasetReleaseValidationError("chk1 override repair population mismatch")
    if root.get("changed_sample_ids_sha256") != _sha256_text(
        _canonical_json(sorted(changed_ids))
    ):
        raise DatasetReleaseValidationError(
            "chk1 override changed sample-ID binding mismatch"
        )

    semantic_input = _required_mapping(
        root.get("semantic_audit_input"), label="semantic_audit_input"
    )
    semantic_input_path = _resolve_release_member(
        candidate_root,
        semantic_input.get("path"),
        label="semantic_audit_input.path",
    )
    if sha256_file(semantic_input_path) != _required_sha256(
        semantic_input.get("sha256"), label="semantic_audit_input.sha256"
    ):
        raise DatasetReleaseValidationError(
            "chk1 override semantic-audit input SHA-256 mismatch"
        )
    if (
        _required_count(semantic_input.get("rows"), label="semantic_audit_input.rows")
        != _CHK1_OVERRIDE_CHANGED_ROWS
        or len(
            _read_jsonl_objects(
                semantic_input_path, label="chk1 override semantic-audit input"
            )
        )
        != _CHK1_OVERRIDE_CHANGED_ROWS
    ):
        raise DatasetReleaseValidationError(
            "chk1 override semantic-audit input row-count mismatch"
        )
    return dict(root), repair_rows


def _verify_chk1_override_validation(
    *,
    report_path: Path,
    expected_report_sha256: str,
    candidate_root: Path,
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_sha256: str,
    repair_rows: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    expected_sha = _required_sha256(
        expected_report_sha256,
        label="dataset_semantic_override_validation_receipt_sha256",
    )
    if sha256_file(report_path) != expected_sha:
        raise DatasetReleaseValidationError(
            "chk1 override deterministic validation receipt SHA-256 mismatch"
        )
    report = _read_json_object(
        report_path, label="chk1 override deterministic validation receipt"
    )
    if (
        report.get("schema_version") != CLEAN_SFT_VALIDATION_SCHEMA
        or report.get("status") != "passed"
    ):
        raise DatasetReleaseValidationError(
            "chk1 override deterministic validation receipt did not pass"
        )
    claimed_digest = _required_sha256(
        report.get("validation_sha256"), label="validation_sha256"
    )
    digest_payload = dict(report)
    digest_payload.pop("validation_sha256", None)
    if claimed_digest != _sha256_text(_canonical_json(digest_payload)):
        raise DatasetReleaseValidationError(
            "chk1 override deterministic validation self-digest mismatch"
        )
    contracts = _required_mapping(report.get("contracts"), label="validation contracts")
    if (
        contracts.get("data") != CLEAN_SFT_DATA_CONTRACT
        or contracts.get("tokenization") != CLEAN_SFT_TOKEN_CONTRACT
    ):
        raise DatasetReleaseValidationError(
            "chk1 override deterministic validation contract mismatch"
        )
    configuration = _required_mapping(
        report.get("configuration"), label="validation configuration"
    )
    if (
        configuration.get("expected_split_counts") != _CHK1_OVERRIDE_SPLIT_COUNTS
        or configuration.get("max_length") != 4608
        or configuration.get("reasoning_token_range") != [512, 2400]
        or configuration.get("answer_token_range") != [16, 512]
    ):
        raise DatasetReleaseValidationError(
            "chk1 override deterministic validation configuration mismatch"
        )
    inputs = _required_mapping(report.get("inputs"), label="validation inputs")
    clean_input = _required_mapping(
        inputs.get("clean_release"), label="validation clean_release"
    )
    if clean_input.get("release_id") != candidate_manifest.get("candidate_id"):
        raise DatasetReleaseValidationError(
            "chk1 override validation candidate identity mismatch"
        )
    manifests = clean_input.get("manifests")
    if (
        not isinstance(manifests, list)
        or sum(
            isinstance(item, Mapping)
            and item.get("name") == "candidate_manifest.json"
            and item.get("sha256") == candidate_manifest_sha256
            for item in manifests
        )
        != 1
    ):
        raise DatasetReleaseValidationError(
            "chk1 override validation candidate manifest binding mismatch"
        )
    clean_splits = _required_mapping(
        clean_input.get("splits"), label="validation clean splits"
    )
    for split in _SPLITS:
        descriptor = _required_mapping(
            clean_splits.get(split), label=f"validation clean split {split}"
        )
        if (
            descriptor.get("sha256")
            != candidate_manifest["split_files"][split]["sha256"]
        ):
            raise DatasetReleaseValidationError(
                f"chk1 override validation {split} split binding mismatch"
            )
    repair_input = _required_mapping(
        inputs.get("repair_manifest"), label="validation repair_manifest"
    )
    repair_descriptor = candidate_manifest["repair_manifest"]
    if repair_input.get("sha256") != repair_descriptor["sha256"] or repair_input.get(
        "rows"
    ) != len(repair_rows):
        raise DatasetReleaseValidationError(
            "chk1 override validation repair-manifest binding mismatch"
        )
    tokenizer = _required_mapping(inputs.get("tokenizer"), label="validation tokenizer")
    tokenizer_files = tokenizer.get("files")
    if not isinstance(tokenizer_files, list) or not tokenizer_files:
        raise DatasetReleaseValidationError(
            "chk1 override validation tokenizer files missing"
        )
    tokenizer_bundle_sha = _required_sha256(
        tokenizer.get("tokenizer_bundle_sha256"), label="validation tokenizer bundle"
    )
    if tokenizer_bundle_sha != _sha256_text(_canonical_json(tokenizer_files)):
        raise DatasetReleaseValidationError(
            "chk1 override validation tokenizer bundle self-digest mismatch"
        )
    counts = _required_mapping(report.get("counts"), label="validation counts")
    total_rows = sum(_CHK1_OVERRIDE_SPLIT_COUNTS.values())
    for key, expected in (
        ("expected_rows", total_rows),
        ("observed_rows", total_rows),
        ("valid_rows", total_rows),
        ("prompt_hashes_unchanged", total_rows),
        ("provided_data_hashes_unchanged", total_rows),
        ("order_unchanged", total_rows),
        ("truncated_rows", 0),
        ("issue_rows_or_groups", 0),
    ):
        if counts.get(key) != expected:
            raise DatasetReleaseValidationError(
                f"chk1 override deterministic validation count mismatch: {key}"
            )
    if counts.get("split_counts") != _CHK1_OVERRIDE_SPLIT_COUNTS:
        raise DatasetReleaseValidationError(
            "chk1 override deterministic validation split-count mismatch"
        )
    gates = _required_mapping(report.get("quality_gates"), label="validation gates")
    if (
        not gates
        or not all(value is True for value in gates.values())
        or report.get("issues") != []
    ):
        raise DatasetReleaseValidationError(
            "chk1 override deterministic validation contains failed gates or issues"
        )
    expected_content_binding = [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "line_number": row["source_line_number"],
            "prompt_sha256": row["prompt_sha256"],
            "provided_data_sha256": row["provided_data_sha256"],
            "response_sha256": row["new_response_sha256"],
        }
        for row in repair_rows
    ]
    if report.get("content_binding_sha256") != _sha256_text(
        _canonical_json(expected_content_binding)
    ):
        raise DatasetReleaseValidationError(
            "chk1 override deterministic validation row-content binding mismatch"
        )
    statistics = _required_mapping(
        report.get("token_statistics"), label="validation token_statistics"
    )
    max_tokens = statistics.get("max_total_tokens")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 0 < max_tokens <= 4608
    ):
        raise DatasetReleaseValidationError(
            "chk1 override deterministic validation max token count is invalid"
        )
    return report


def _verify_chk1_override_failed_audit(
    *,
    summary_path: Path,
    expected_summary_sha256: str,
    candidate_root: Path,
    candidate_manifest: Mapping[str, Any],
    repair_rows: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    expected_sha = _required_sha256(
        expected_summary_sha256,
        label="dataset_semantic_override_audit_summary_sha256",
    )
    if sha256_file(summary_path) != expected_sha:
        raise DatasetReleaseValidationError(
            "chk1 override failed semantic-audit summary SHA-256 mismatch"
        )
    summary = _read_json_object(
        summary_path, label="chk1 override semantic-audit summary"
    )
    if (
        summary.get("schema_version") != CLEAN_SFT_AUDIT_SCHEMA
        or summary.get("status") != "failed"
    ):
        raise DatasetReleaseValidationError(
            "chk1 override requires a failed source-only semantic-audit summary"
        )
    try:
        release_dir = _canonical_path(
            str(summary.get("release_dir") or ""),
            label="semantic audit release_dir",
            directory=True,
        )
    except DatasetReleaseValidationError:
        raise
    if release_dir != candidate_root:
        raise DatasetReleaseValidationError(
            "chk1 override semantic audit is bound to a different candidate"
        )
    if (
        summary.get("repair_manifest_sha256")
        != candidate_manifest["repair_manifest"]["sha256"]
    ):
        raise DatasetReleaseValidationError(
            "chk1 override semantic audit repair-manifest binding mismatch"
        )
    counts = _required_mapping(summary.get("counts"), label="semantic audit counts")
    expected = _required_count(counts.get("expected"), label="audit counts.expected")
    completed = _required_count(counts.get("completed"), label="audit counts.completed")
    passed = _required_count(counts.get("passed"), label="audit counts.passed")
    failed = _required_count(counts.get("failed"), label="audit counts.failed")
    judge_errors = _required_count(
        counts.get("judge_errors"), label="audit counts.judge_errors"
    )
    blocking = _required_count(
        counts.get("blocking_violations"), label="audit counts.blocking_violations"
    )
    errors = summary.get("errors")
    if (
        expected != _CHK1_OVERRIDE_CHANGED_ROWS
        or completed + judge_errors != expected
        or passed + failed != completed
        or not isinstance(errors, list)
        or len(errors) != judge_errors
        or (failed == 0 and judge_errors == 0 and blocking == 0)
    ):
        raise DatasetReleaseValidationError(
            "chk1 override failed semantic-audit counts are inconsistent"
        )
    row_descriptor = _required_mapping(
        summary.get("row_audit"), label="semantic audit row_audit"
    )
    row_path = _resolve_release_member(
        summary_path.parent,
        row_descriptor.get("path"),
        label="semantic audit row_audit.path",
    )
    if sha256_file(row_path) != _required_sha256(
        row_descriptor.get("sha256"), label="semantic audit row_audit.sha256"
    ):
        raise DatasetReleaseValidationError(
            "chk1 override semantic row-audit SHA-256 mismatch"
        )
    if (
        _required_count(
            row_descriptor.get("rows"), label="semantic audit row_audit.rows"
        )
        != completed
    ):
        raise DatasetReleaseValidationError(
            "chk1 override semantic row-audit row-count mismatch"
        )
    audit_rows = _read_jsonl_objects(row_path, label="semantic row audits")
    if len(audit_rows) != completed:
        raise DatasetReleaseValidationError(
            "chk1 override semantic row-audit physical row-count mismatch"
        )
    expected_changed = {
        str(row["sample_id"]): (
            str(row["split"]),
            str(row["new_response_sha256"]),
            str(row["provided_data_sha256"]),
        )
        for row in repair_rows
        if row["old_response_sha256"] != row["new_response_sha256"]
    }
    observed_ids: set[str] = set()
    observed_passed = 0
    observed_failed = 0
    observed_blocking = 0
    for number, row in enumerate(audit_rows, 1):
        sample_id = row.get("sample_id")
        split = row.get("split")
        status = row.get("status")
        blockers = row.get("blocking_violations")
        validated = row.get("validated_violations")
        if (
            row.get("schema_version") != CLEAN_SFT_AUDIT_SCHEMA
            or not isinstance(sample_id, str)
            or sample_id not in expected_changed
            or sample_id in observed_ids
            or split != expected_changed[sample_id][0]
            or status not in {"passed", "failed"}
            or not isinstance(blockers, list)
            or not isinstance(validated, list)
            or (status == "passed" and blockers)
            or (status == "failed" and not blockers)
        ):
            raise DatasetReleaseValidationError(
                f"chk1 override semantic row-audit row {number} is invalid"
            )
        if (
            _required_sha256(
                row.get("candidate_sha256"),
                label=f"semantic row {number}.candidate_sha256",
            )
            != expected_changed[sample_id][1]
            or _required_sha256(
                row.get("evidence_sha256"),
                label=f"semantic row {number}.evidence_sha256",
            )
            != expected_changed[sample_id][2]
        ):
            raise DatasetReleaseValidationError(
                f"chk1 override semantic row-audit row {number} content binding mismatch"
            )
        observed_ids.add(sample_id)
        observed_passed += int(status == "passed")
        observed_failed += int(status == "failed")
        observed_blocking += len(blockers)
    if (
        observed_passed != passed
        or observed_failed != failed
        or observed_blocking != blocking
    ):
        raise DatasetReleaseValidationError(
            "chk1 override semantic row-audit aggregates disagree with summary"
        )
    error_ids: set[str] = set()
    for number, error in enumerate(errors, 1):
        if not isinstance(error, Mapping):
            raise DatasetReleaseValidationError(
                f"chk1 override semantic audit error {number} is invalid"
            )
        sample_id = error.get("sample_id")
        split = error.get("split")
        if (
            not isinstance(sample_id, str)
            or sample_id not in expected_changed
            or sample_id in observed_ids
            or sample_id in error_ids
            or split != expected_changed[sample_id][0]
        ):
            raise DatasetReleaseValidationError(
                f"chk1 override semantic audit error {number} identity mismatch"
            )
        error_ids.add(sample_id)
    if observed_ids | error_ids != set(expected_changed):
        raise DatasetReleaseValidationError(
            "chk1 override semantic audit does not cover every changed sample"
        )
    judge = _required_mapping(summary.get("judge"), label="semantic audit judge")
    health = _required_mapping(judge.get("health"), label="semantic audit judge.health")
    if (
        judge.get("model") != _JUDGE_MODEL
        or health.get("model") != _JUDGE_MODEL
        or Path(str(health.get("loaded_model_root") or "")).name != _JUDGE_MODEL
        or health.get("status") != "ready"
        or health.get("tokenizer_parity") is not True
        or health.get("weight_attested") is not True
    ):
        raise DatasetReleaseValidationError(
            "chk1 override semantic-audit judge health is not attested"
        )
    return summary


def _verify_chk1_override_authorization(
    *,
    receipt_path: Path,
    expected_receipt_sha256: str,
    dataset: Path,
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_sha256: str,
    validation_receipt_sha256: str,
    audit_summary_sha256: str,
    training_stage: str,
) -> Mapping[str, Any]:
    expected_sha = _required_sha256(
        expected_receipt_sha256,
        label="dataset_semantic_override_authorization_receipt_sha256",
    )
    if sha256_file(receipt_path) != expected_sha:
        raise DatasetReleaseValidationError(
            "chk1 override authorization receipt SHA-256 mismatch"
        )
    receipt = _read_json_object(
        receipt_path, label="chk1 override authorization receipt"
    )
    if (
        receipt.get("schema_version") != CHK1_OVERRIDE_AUTHORIZATION_SCHEMA
        or receipt.get("status") != "authorized"
        or receipt.get("authorization_basis") != "explicit_user_instruction"
    ):
        raise DatasetReleaseValidationError("chk1 override authorization is invalid")
    claimed_digest = _required_sha256(
        receipt.get("authorization_sha256"), label="authorization_sha256"
    )
    digest_payload = dict(receipt)
    digest_payload.pop("authorization_sha256", None)
    if claimed_digest != _sha256_text(_canonical_json(digest_payload)):
        raise DatasetReleaseValidationError(
            "chk1 override authorization self-digest mismatch"
        )
    scope = _required_mapping(receipt.get("scope"), label="authorization scope")
    if (
        training_stage != "chk1"
        or scope.get("stage") != "chk1"
        or scope.get("operation") != "sft_training"
        or scope.get("downstream_stages_allowed") != []
    ):
        raise DatasetReleaseValidationError(
            "semantic override is restricted to chk1 SFT and prohibits chk2/chk3/chk4"
        )
    bindings = _required_mapping(
        receipt.get("bindings"), label="authorization bindings"
    )
    if (
        bindings.get("candidate_id") != candidate_manifest.get("candidate_id")
        or bindings.get("dataset_dir") != str(dataset)
        or bindings.get("candidate_manifest_sha256") != candidate_manifest_sha256
        or bindings.get("deterministic_validation_receipt_sha256")
        != validation_receipt_sha256
        or bindings.get("semantic_audit_summary_sha256") != audit_summary_sha256
        or bindings.get("split_counts") != _CHK1_OVERRIDE_SPLIT_COUNTS
    ):
        raise DatasetReleaseValidationError(
            "chk1 override authorization does not bind all audited inputs"
        )
    acknowledgements = receipt.get("risk_acknowledgements")
    if (
        not isinstance(acknowledgements, list)
        or len(acknowledgements) != len(_CHK1_OVERRIDE_ACKNOWLEDGEMENTS)
        or set(acknowledgements) != _CHK1_OVERRIDE_ACKNOWLEDGEMENTS
    ):
        raise DatasetReleaseValidationError(
            "chk1 override authorization lacks the required risk acknowledgements"
        )
    return receipt


def verify_chk1_semantic_override(
    *,
    dataset_dir: str | Path,
    candidate_manifest_path: str | Path,
    expected_candidate_manifest_sha256: str,
    deterministic_validation_path: str | Path,
    expected_deterministic_validation_sha256: str,
    semantic_audit_summary_path: str | Path,
    expected_semantic_audit_summary_sha256: str,
    authorization_receipt_path: str | Path,
    expected_authorization_receipt_sha256: str,
    training_stage: str,
) -> dict[str, Any]:
    """Verify an explicit, hash-bound exception for chk1 SFT only.

    This verifier intentionally accepts a *failed* semantic audit, but it does
    not waive the immutable-candidate, deterministic data/token validation, or
    exact split-file contracts.  The authorization receipt is self-digested,
    pinned by the training config, and explicitly prohibits downstream use.
    """

    dataset = _canonical_path(dataset_dir, label="dataset_name", directory=True)
    candidate_manifest_file = _canonical_path(
        candidate_manifest_path,
        label="dataset_semantic_override_candidate_manifest",
        directory=False,
    )
    validation_file = _canonical_path(
        deterministic_validation_path,
        label="dataset_semantic_override_validation_receipt",
        directory=False,
    )
    audit_file = _canonical_path(
        semantic_audit_summary_path,
        label="dataset_semantic_override_audit_summary",
        directory=False,
    )
    authorization_file = _canonical_path(
        authorization_receipt_path,
        label="dataset_semantic_override_authorization_receipt",
        directory=False,
    )
    candidate_manifest, repair_rows = _verify_chk1_override_candidate(
        dataset=dataset,
        manifest=candidate_manifest_file,
        expected_manifest_sha256=expected_candidate_manifest_sha256,
    )
    candidate_sha = _required_sha256(
        expected_candidate_manifest_sha256,
        label="dataset_semantic_override_candidate_manifest_sha256",
    )
    validation_sha = _required_sha256(
        expected_deterministic_validation_sha256,
        label="dataset_semantic_override_validation_receipt_sha256",
    )
    audit_sha = _required_sha256(
        expected_semantic_audit_summary_sha256,
        label="dataset_semantic_override_audit_summary_sha256",
    )
    _verify_chk1_override_validation(
        report_path=validation_file,
        expected_report_sha256=validation_sha,
        candidate_root=candidate_manifest_file.parent,
        candidate_manifest=candidate_manifest,
        candidate_manifest_sha256=candidate_sha,
        repair_rows=repair_rows,
    )
    audit = _verify_chk1_override_failed_audit(
        summary_path=audit_file,
        expected_summary_sha256=audit_sha,
        candidate_root=candidate_manifest_file.parent,
        candidate_manifest=candidate_manifest,
        repair_rows=repair_rows,
    )
    authorization = _verify_chk1_override_authorization(
        receipt_path=authorization_file,
        expected_receipt_sha256=expected_authorization_receipt_sha256,
        dataset=dataset,
        candidate_manifest=candidate_manifest,
        candidate_manifest_sha256=candidate_sha,
        validation_receipt_sha256=validation_sha,
        audit_summary_sha256=audit_sha,
        training_stage=training_stage,
    )
    return {
        "mode": "chk1_semantic_override",
        "release_id": candidate_manifest["candidate_id"],
        "quality_status": candidate_manifest["quality_status"],
        "split_files": dict(candidate_manifest["split_files"]),
        "scope": dict(authorization["scope"]),
        "bindings": {
            "candidate_manifest_sha256": candidate_sha,
            "deterministic_validation_receipt_sha256": validation_sha,
            "semantic_audit_summary_sha256": audit_sha,
            "authorization_receipt_sha256": _required_sha256(
                expected_authorization_receipt_sha256,
                label="dataset_semantic_override_authorization_receipt_sha256",
            ),
            "semantic_audit_counts": dict(audit["counts"]),
        },
    }


__all__ = [
    "CHK1_OVERRIDE_AUTHORIZATION_SCHEMA",
    "CHK1_OVERRIDE_CANDIDATE_SCHEMA",
    "CLEAN_SFT_RELEASE_SCHEMA",
    "CLEAN_SFT_AUDIT_SCHEMA",
    "CLEAN_SFT_VALIDATION_SCHEMA",
    "CHK4_PRE2009_AUGMENTED_RELEASE_ID",
    "CHK4_PRE2009_AUGMENTED_RELEASE_SCHEMA",
    "CHK4_PRE2009_CORRECTION_RELEASE_ID",
    "CHK4_PRE2009_CORRECTION_RELEASE_SCHEMA",
    "CHK4_PRE2009_CORRECTION_SAMPLER_TYPE",
    "CHK4_PRE2009_CORRECTION_SFT_ROLE",
    "CHK4_PRE2009_GRPO_ROLE",
    "CHK4_PRE2009_ROLES",
    "CHK4_PRE2009_SFT_ROLE",
    "DatasetReleaseValidationError",
    "PAPER_CHK2_BINDING_SCHEMA",
    "PAPER_CHK2_DATASET_ROLE",
    "PAPER_CHK2_HANDOFF_SCHEMA",
    "PAPER_CHK2_PARENT_MODEL_SHA256",
    "PAPER_CHK2_RELEASE_SCHEMA",
    "PAPER_CHK2_STUDENT_SYSTEM_PROMPT",
    "PAPER_CHK2_STUDENT_SYSTEM_PROMPT_SHA256",
    "PAPER_CHK2_TRAINING_SCOPE",
    "PAPER_CHK2_USER_PROMPT_TEMPLATE_SHA256",
    "STANDALONE_CHK3_BINDING_SCHEMA",
    "STANDALONE_CHK3_DATASET_ROLE",
    "STANDALONE_CHK3_DIRECT_SCOPE",
    "STANDALONE_CHK3_RELEASE_SCHEMA",
    "sha256_file",
    "verify_chk1_semantic_override",
    "verify_clean_sft_release",
    "verify_paper_chk2_sft_release",
    "verify_chk4_pre2009_augmented_release",
    "verify_chk4_pre2009_correction_release",
    "verify_chk4_release_for_role",
    "verify_standalone_chk3_direct_sft_release",
]
