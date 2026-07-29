"""Validation for token-length-matched neutral LOO prompt interventions.

This module is intentionally independent of generation jobs.  It validates a
``loo-neutral-intervention-v1`` draft manifest and its prompt directory, then
returns a normalized manifest with a complete ``input_artifacts`` inventory
that a generation runner can freeze into its own run manifest.
"""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Protocol

from open_r1.provenance import sha256_file, sha256_text
from open_r1.validator.intervention import (
    has_line_block_boundaries,
    match_indicator_marker,
    normalise_indicator_text,
)


NEUTRAL_MANIFEST_SCHEMA_VERSION = "loo-neutral-intervention-v1"
ROSTER_SCHEMA_VERSION = "loo-intervention-roster-v1"

FORBIDDEN_PROMPT_ROW_FIELDS = frozenset(
    {
        "actual_minutes",
        "actual_output",
        "answer",
        "gold",
        "gold_text",
        "ground_truth",
        "minutes",
        "reference",
        "reference_text",
        "target",
        "target_text",
    }
)


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ...


def _load_json_object(
    value: str | Path | Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _load_roster(path: Path) -> dict[str, Any]:
    roster = _load_json_object(path, label="neutral intervention roster")
    if roster.get("schema_version") != ROSTER_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported neutral intervention roster schema: "
            f"{roster.get('schema_version')!r}"
        )
    roster_id = str(roster.get("roster_id") or path.stem).strip()
    baseline = str(roster.get("baseline_indicator") or "").strip()
    indicators = [str(value).strip() for value in roster.get("indicators", [])]
    contexts = [str(value).strip() for value in roster.get("contexts", [])]
    if not roster_id or not baseline or not indicators or not contexts:
        raise ValueError(
            "Neutral intervention roster requires roster_id, baseline_indicator, "
            "indicators, and contexts"
        )
    if (
        any(not value for value in indicators)
        or any(not value for value in contexts)
        or len(indicators) != len(set(indicators))
        or len(contexts) != len(set(contexts))
    ):
        raise ValueError("Neutral intervention roster has blank or duplicate values")
    if baseline in indicators:
        raise ValueError("baseline_indicator must not appear in indicators")

    raw_markers = roster.get("indicator_markers")
    if not isinstance(raw_markers, dict) or set(raw_markers) != set(indicators):
        raise ValueError(
            "Neutral intervention roster requires marker lists for exactly the "
            "declared indicators"
        )
    markers: dict[str, list[str]] = {}
    for indicator in indicators:
        values = raw_markers[indicator]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Roster has no markers for indicator {indicator!r}")
        marker_values = [str(value).strip() for value in values]
        if any(not normalise_indicator_text(value) for value in marker_values):
            raise ValueError(f"Roster has an invalid marker for {indicator!r}")
        markers[indicator] = marker_values
    return {
        "schema_version": ROSTER_SCHEMA_VERSION,
        "roster_id": roster_id,
        "baseline_indicator": baseline,
        "indicators": indicators,
        "contexts": contexts,
        "indicator_markers": markers,
    }


def _parse_artifact_identity(path: Path) -> tuple[str, str]:
    if "_masked_" not in path.stem:
        raise ValueError(
            f"Neutral prompt filename {path.name!r} must use "
            "'<indicator>_masked_<context>.jsonl'"
        )
    indicator, context = path.stem.split("_masked_", 1)
    if not indicator.strip() or not context.strip():
        raise ValueError(f"Incomplete neutral prompt identity in {path.name!r}")
    return indicator.strip(), context.strip()


def _canonical_meeting_date(value: object, *, location: str) -> str:
    text = str(value or "").strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"Neutral prompt row has invalid meeting_date at {location}: {value!r}"
        ) from exc
    if parsed.isoformat() != text:
        raise ValueError(
            f"Neutral prompt meeting_date is not canonical at {location}: {value!r}"
        )
    return text


def _read_prompt_rows(
    path: Path,
    *,
    indicator: str,
    context: str,
    baseline_indicator: str,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid neutral prompt JSON in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Neutral prompt row must be an object in {path}:{line_number}"
                )
            forbidden = sorted(FORBIDDEN_PROMPT_ROW_FIELDS & set(row))
            if forbidden:
                raise ValueError(
                    f"Neutral prompt row {path}:{line_number} contains prohibited "
                    f"target/reference fields: {forbidden}"
                )
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Neutral prompt is empty in {path}:{line_number}")
            sample_id = str(row.get("sample_id") or "").strip()
            meeting_date = _canonical_meeting_date(
                row.get("meeting_date"),
                location=f"{path}:{line_number}",
            )
            section_name = str(row.get("section_name") or "").strip()
            if not sample_id or not section_name:
                raise ValueError(
                    f"Neutral prompt row {path}:{line_number} requires sample_id "
                    "and section_name"
                )
            expected_sample_id = f"{meeting_date}::{section_name}"
            if sample_id != expected_sample_id:
                raise ValueError(
                    f"Neutral prompt sample_id mismatch in {path}:{line_number}: "
                    f"expected {expected_sample_id!r}, observed {sample_id!r}"
                )
            if sample_id in rows:
                raise ValueError(f"Duplicate neutral prompt sample_id in {path}: {sample_id}")

            expected_arm = "full" if indicator == baseline_indicator else "neutral"
            expected_metadata = {
                "indicator": indicator,
                "evaluation_context": context,
                "arm": expected_arm,
            }
            mismatches = {
                field: {"expected": expected, "observed": row.get(field)}
                for field, expected in expected_metadata.items()
                if row.get(field) != expected
            }
            if mismatches:
                raise ValueError(
                    f"Neutral prompt row metadata mismatch in {path}:{line_number}: "
                    f"{mismatches}"
                )
            declared_hash = str(row.get("prompt_sha256") or "").strip()
            observed_hash = sha256_text(prompt)
            if declared_hash != observed_hash:
                raise ValueError(
                    f"Neutral prompt row hash mismatch in {path}:{line_number}: "
                    f"declared={declared_hash!r}, observed={observed_hash!r}"
                )
            rows[sample_id] = row
    if not rows:
        raise ValueError(f"Neutral prompt artifact is empty: {path}")
    return rows


def _token_count(tokenizer: TokenizerLike, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(token_ids, (list, tuple)):
        raise TypeError("Tokenizer.encode must return a list or tuple of token IDs")
    return len(token_ids)


def _required_int(proof: Mapping[str, Any], field: str, *, identity: str) -> int:
    value = proof.get(field)
    if isinstance(value, bool):
        raise ValueError(f"Neutral proof {identity} has invalid {field}: {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Neutral proof {identity} requires integer {field}, got {value!r}"
        ) from exc
    if value != parsed:
        raise ValueError(
            f"Neutral proof {identity} requires canonical integer {field}, "
            f"got {value!r}"
        )
    return parsed


def _proof_context(
    proof: Mapping[str, Any],
    *,
    contexts: list[str],
    manifest: Mapping[str, Any],
) -> str:
    declared = str(proof.get("context") or "").strip()
    if declared:
        return declared
    manifest_context = str(manifest.get("population_context") or "").strip()
    if manifest_context:
        return manifest_context
    if len(contexts) == 1:
        return contexts[0]
    raise ValueError(
        "Neutral replacement proof requires context when the roster has multiple "
        "contexts"
    )


def _validate_preserved_indicator_frame(
    *,
    source_block: str,
    neutral_block: str,
    indicator: str,
    markers: list[str],
) -> None:
    analysis_delimiter = "Analysis:\n"
    if (
        analysis_delimiter not in source_block
        or analysis_delimiter not in neutral_block
    ):
        raise ValueError(
            f"Neutral replacement for {indicator!r} lacks the Analysis delimiter"
        )
    source_header, source_tail = source_block.split(analysis_delimiter, 1)
    neutral_header, neutral_tail = neutral_block.split(analysis_delimiter, 1)
    if source_header != neutral_header:
        raise ValueError(
            f"Neutral replacement for {indicator!r} changed its indicator label/header"
        )

    footer = f"<<<END-LOO-INDICATOR-BLOCK:{indicator}>>>\n"
    if not source_tail.endswith(footer) or not neutral_tail.endswith(footer):
        raise ValueError(
            f"Neutral replacement for {indicator!r} changed or lacks its delimiter"
        )
    if match_indicator_marker(source_block, markers) is None:
        raise ValueError(
            f"Source block for {indicator!r} does not contain a declared marker "
            "in its first semantic line"
        )
    if match_indicator_marker(neutral_block, markers) is None:
        raise ValueError(
            f"Neutral block for {indicator!r} does not preserve a declared marker "
            "in its first semantic line"
        )

    source_body = source_tail[: -len(footer)]
    neutral_body = neutral_tail[: -len(footer)]
    if not source_body.strip() or not neutral_body.strip():
        raise ValueError(
            f"Neutral replacement for {indicator!r} has an empty source or neutral body"
        )
    if source_body == neutral_body:
        raise ValueError(
            f"Neutral replacement for {indicator!r} did not replace the analysis body"
        )


def _validate_declared_input_artifacts(
    declared: object,
    *,
    observed: list[dict[str, Any]],
) -> None:
    if declared is None:
        return
    if not isinstance(declared, list):
        raise ValueError("neutral manifest input_artifacts must be a list")
    expected_by_identity = {
        (entry["indicator"], entry["context"]): entry for entry in observed
    }
    declared_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in declared:
        if not isinstance(entry, Mapping):
            raise ValueError("neutral manifest input_artifacts entries must be objects")
        identity = (
            str(entry.get("indicator") or "").strip(),
            str(entry.get("context") or "").strip(),
        )
        if identity in declared_by_identity:
            raise ValueError(f"Duplicate neutral input_artifact identity: {identity}")
        declared_by_identity[identity] = entry
    if set(declared_by_identity) != set(expected_by_identity):
        raise ValueError(
            "Neutral manifest input_artifacts do not match the complete artifact "
            "universe"
        )
    for identity, expected in expected_by_identity.items():
        entry = declared_by_identity[identity]
        mismatches = {
            field: {"expected": expected[field], "observed": entry.get(field)}
            for field in (
                "relative_path",
                "indicator",
                "context",
                "source_row_count",
                "source_file_sha256",
            )
            if entry.get(field) != expected[field]
        }
        if mismatches:
            raise ValueError(
                f"Neutral manifest input_artifact mismatch for {identity}: "
                f"{mismatches}"
            )


def validate_neutral_intervention_manifest(
    *,
    manifest: str | Path | Mapping[str, Any],
    input_dir: str | Path,
    roster_file: str | Path,
    tokenizer: TokenizerLike,
    tokenizer_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize a neutral-intervention manifest.

    Args:
        manifest: Draft manifest mapping or path to its JSON file.
        input_dir: Directory containing one baseline and one neutral JSONL file
            per indicator/context, named
            ``<indicator>_masked_<context>.jsonl``.
        roster_file: Frozen ``loo-intervention-roster-v1`` used for the run.
        tokenizer: The exact tokenizer used to construct the neutral prompts.
        tokenizer_artifact_sha256: Optional frozen tokenizer artifact digest.
            When supplied, it must equal ``manifest.tokenizer_artifact.sha256``.

    Returns:
        A deep-copied, normalized manifest containing validated, generation-ready
        ``input_artifacts``, normalized replacement proofs, and an ``audit``
        object.  The caller may freeze this return value directly.
    """

    draft = _load_json_object(manifest, label="neutral intervention manifest")
    if draft.get("schema_version") != NEUTRAL_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported neutral intervention manifest schema: "
            f"{draft.get('schema_version')!r}"
        )
    folder = Path(input_dir).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Neutral prompt directory does not exist: {folder}")
    roster_path = Path(roster_file).expanduser().resolve()
    roster = _load_roster(roster_path)

    expected_manifest_metadata = {
        "roster_id": roster["roster_id"],
        "roster_file_sha256": sha256_file(roster_path),
        "baseline_indicator": roster["baseline_indicator"],
        "indicators": roster["indicators"],
        "contexts": roster["contexts"],
    }
    metadata_mismatches = {
        field: {"expected": expected, "observed": draft.get(field)}
        for field, expected in expected_manifest_metadata.items()
        if draft.get(field) != expected
    }
    if metadata_mismatches:
        raise ValueError(
            f"Neutral manifest/roster metadata mismatch: {metadata_mismatches}"
        )

    if tokenizer_artifact_sha256 is not None:
        expected_tokenizer_hash = str(tokenizer_artifact_sha256).strip().lower()
        if (
            len(expected_tokenizer_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_tokenizer_hash
            )
        ):
            raise ValueError(
                "tokenizer_artifact_sha256 must be a lowercase 64-character digest"
            )
        tokenizer_metadata = draft.get("tokenizer_artifact")
        observed_tokenizer_hash = (
            str(tokenizer_metadata.get("sha256") or "").strip()
            if isinstance(tokenizer_metadata, Mapping)
            else ""
        )
        if observed_tokenizer_hash != expected_tokenizer_hash:
            raise ValueError(
                "Neutral manifest tokenizer artifact hash mismatch: "
                f"expected={expected_tokenizer_hash}, "
                f"observed={observed_tokenizer_hash!r}"
            )

    expected_identities = {
        (indicator, context)
        for context in roster["contexts"]
        for indicator in [
            roster["baseline_indicator"],
            *roster["indicators"],
        ]
    }
    paths_by_identity: dict[tuple[str, str], Path] = {}
    for path in sorted(folder.glob("*.jsonl")):
        identity = _parse_artifact_identity(path)
        if identity in paths_by_identity:
            raise ValueError(f"Duplicate neutral prompt artifact identity: {identity}")
        paths_by_identity[identity] = path
    if set(paths_by_identity) != expected_identities:
        missing = sorted(expected_identities - set(paths_by_identity))
        extra = sorted(set(paths_by_identity) - expected_identities)
        raise ValueError(
            "Neutral prompt artifacts do not match the exact roster universe: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    rows_by_identity = {
        identity: _read_prompt_rows(
            path,
            indicator=identity[0],
            context=identity[1],
            baseline_indicator=roster["baseline_indicator"],
        )
        for identity, path in paths_by_identity.items()
    }
    input_artifacts: list[dict[str, Any]] = []
    for context in roster["contexts"]:
        for indicator in [
            roster["baseline_indicator"],
            *roster["indicators"],
        ]:
            identity = (indicator, context)
            path = paths_by_identity[identity]
            input_artifacts.append(
                {
                    "relative_path": path.relative_to(folder).as_posix(),
                    "indicator": indicator,
                    "context": context,
                    "arm": (
                        "full"
                        if indicator == roster["baseline_indicator"]
                        else "neutral"
                    ),
                    "source_row_count": len(rows_by_identity[identity]),
                    "source_file_sha256": sha256_file(path),
                }
            )
    _validate_declared_input_artifacts(
        draft.get("input_artifacts"),
        observed=input_artifacts,
    )

    expected_proof_keys: set[tuple[str, str, str]] = set()
    for context in roster["contexts"]:
        baseline_rows = rows_by_identity[(roster["baseline_indicator"], context)]
        baseline_keys = set(baseline_rows)
        neutral_hashes_by_sample: dict[str, dict[str, str]] = {
            sample_id: {} for sample_id in baseline_keys
        }
        for indicator in roster["indicators"]:
            neutral_rows = rows_by_identity[(indicator, context)]
            if set(neutral_rows) != baseline_keys:
                missing = sorted(baseline_keys - set(neutral_rows))
                extra = sorted(set(neutral_rows) - baseline_keys)
                raise ValueError(
                    f"Neutral prompt row coverage mismatch for "
                    f"{indicator}/{context}: missing={missing[:5]}, "
                    f"extra={extra[:5]}"
                )
            for sample_id in sorted(baseline_keys):
                full_row = baseline_rows[sample_id]
                neutral_row = neutral_rows[sample_id]
                for field in ("meeting_date", "section_name", "section_family"):
                    if (
                        field in full_row
                        or field in neutral_row
                    ) and full_row.get(field) != neutral_row.get(field):
                        raise ValueError(
                            f"Neutral prompt identity metadata mismatch for "
                            f"{indicator}/{context}/{sample_id}: {field}"
                        )
                neutral_hash = sha256_text(str(neutral_row["prompt"]))
                if neutral_hash == sha256_text(str(full_row["prompt"])):
                    raise ValueError(
                        f"Neutral prompt is identical to baseline for "
                        f"{indicator}/{context}/{sample_id}"
                    )
                previous = neutral_hashes_by_sample[sample_id].get(neutral_hash)
                if previous is not None:
                    raise ValueError(
                        f"Neutral prompts are not distinct for "
                        f"{context}/{sample_id}: indicators {previous!r} and "
                        f"{indicator!r}"
                    )
                neutral_hashes_by_sample[sample_id][neutral_hash] = indicator
                expected_proof_keys.add((context, indicator, sample_id))

    raw_replacements = draft.get("replacements")
    if not isinstance(raw_replacements, list):
        raise ValueError("Neutral manifest requires a replacements list")
    proofs_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw_proof in raw_replacements:
        if not isinstance(raw_proof, dict):
            raise ValueError("Neutral replacement proofs must be objects")
        context = _proof_context(
            raw_proof,
            contexts=roster["contexts"],
            manifest=draft,
        )
        indicator = str(raw_proof.get("indicator") or "").strip()
        sample_id = str(raw_proof.get("sample_id") or "").strip()
        key = (context, indicator, sample_id)
        if key in proofs_by_key:
            raise ValueError(f"Duplicate neutral replacement proof: {key}")
        normalized_proof = copy.deepcopy(raw_proof)
        normalized_proof["context"] = context
        proofs_by_key[key] = normalized_proof
    if set(proofs_by_key) != expected_proof_keys:
        missing = sorted(expected_proof_keys - set(proofs_by_key))
        extra = sorted(set(proofs_by_key) - expected_proof_keys)
        raise ValueError(
            "Neutral replacement proofs do not cover the exact row universe: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    normalized_proofs: list[dict[str, Any]] = []
    for context in roster["contexts"]:
        baseline_rows = rows_by_identity[(roster["baseline_indicator"], context)]
        for indicator in roster["indicators"]:
            neutral_rows = rows_by_identity[(indicator, context)]
            for sample_id in sorted(baseline_rows):
                identity = f"{context}/{indicator}/{sample_id}"
                proof = proofs_by_key[(context, indicator, sample_id)]
                full_row = baseline_rows[sample_id]
                neutral_row = neutral_rows[sample_id]
                full_prompt = str(full_row["prompt"])
                neutral_prompt = str(neutral_row["prompt"])

                expected_identity_metadata = {
                    "meeting_date": full_row["meeting_date"],
                    "section_family": full_row.get(
                        "section_family",
                        full_row["section_name"],
                    ),
                    "full_prompt_sha256": sha256_text(full_prompt),
                    "neutral_prompt_sha256": sha256_text(neutral_prompt),
                }
                proof_mismatches = {
                    field: {"expected": expected, "observed": proof.get(field)}
                    for field, expected in expected_identity_metadata.items()
                    if proof.get(field) != expected
                }
                if proof_mismatches:
                    raise ValueError(
                        f"Neutral proof identity/hash mismatch for {identity}: "
                        f"{proof_mismatches}"
                    )

                start = _required_int(proof, "replacement_start", identity=identity)
                source_end = _required_int(
                    proof,
                    "source_replacement_end",
                    identity=identity,
                )
                neutral_end = _required_int(
                    proof,
                    "neutral_replacement_end",
                    identity=identity,
                )
                if (
                    not 0 <= start < source_end <= len(full_prompt)
                    or not 0 <= start < neutral_end <= len(neutral_prompt)
                ):
                    raise ValueError(f"Neutral proof span is out of bounds for {identity}")
                source_prefix = full_prompt[:start]
                neutral_prefix = neutral_prompt[:start]
                source_suffix = full_prompt[source_end:]
                neutral_suffix = neutral_prompt[neutral_end:]
                if source_prefix != neutral_prefix or source_suffix != neutral_suffix:
                    raise ValueError(
                        f"Neutral proof prefix/suffix changed outside the one "
                        f"replacement span for {identity}"
                    )
                if not has_line_block_boundaries(full_prompt, start, source_end):
                    raise ValueError(
                        f"Source replacement is not line/block bounded for {identity}"
                    )
                if not has_line_block_boundaries(neutral_prompt, start, neutral_end):
                    raise ValueError(
                        f"Neutral replacement is not line/block bounded for {identity}"
                    )

                source_block = full_prompt[start:source_end]
                neutral_block = neutral_prompt[start:neutral_end]
                if source_block == neutral_block:
                    raise ValueError(f"Neutral proof contains no replacement for {identity}")
                hash_expectations = {
                    "source_block_sha256": sha256_text(source_block),
                    "neutral_block_sha256": sha256_text(neutral_block),
                    "prefix_sha256": sha256_text(source_prefix),
                    "suffix_sha256": sha256_text(source_suffix),
                }
                hash_mismatches = {
                    field: {"expected": expected, "observed": proof.get(field)}
                    for field, expected in hash_expectations.items()
                    if proof.get(field) != expected
                }
                if hash_mismatches:
                    raise ValueError(
                        f"Neutral proof block/prefix/suffix hash mismatch for "
                        f"{identity}: {hash_mismatches}"
                    )

                _validate_preserved_indicator_frame(
                    source_block=source_block,
                    neutral_block=neutral_block,
                    indicator=indicator,
                    markers=roster["indicator_markers"][indicator],
                )
                source_block_tokens = _token_count(tokenizer, source_block)
                neutral_block_tokens = _token_count(tokenizer, neutral_block)
                source_full_tokens = _token_count(tokenizer, full_prompt)
                neutral_full_tokens = _token_count(tokenizer, neutral_prompt)
                if source_block_tokens != neutral_block_tokens:
                    raise ValueError(
                        f"Neutral block token-count mismatch for {identity}: "
                        f"{source_block_tokens} != {neutral_block_tokens}"
                    )
                if source_full_tokens != neutral_full_tokens:
                    raise ValueError(
                        f"Neutral full-prompt token-count mismatch for {identity}: "
                        f"{source_full_tokens} != {neutral_full_tokens}"
                    )
                declared_token_counts = {
                    "source_block_token_count_no_special_tokens": source_block_tokens,
                    "neutral_block_token_count_no_special_tokens": neutral_block_tokens,
                    "source_full_prompt_token_count_no_special_tokens": (
                        source_full_tokens
                    ),
                    "neutral_full_prompt_token_count_no_special_tokens": (
                        neutral_full_tokens
                    ),
                }
                count_mismatches = {
                    field: {"expected": expected, "observed": proof.get(field)}
                    for field, expected in declared_token_counts.items()
                    if proof.get(field) != expected
                }
                if count_mismatches:
                    raise ValueError(
                        f"Neutral proof declared token-count mismatch for "
                        f"{identity}: {count_mismatches}"
                    )
                for field in (
                    "label_and_delimiters_preserved",
                    "prefix_suffix_unchanged",
                    "token_length_matched",
                ):
                    if proof.get(field) is not True:
                        raise ValueError(
                            f"Neutral proof {identity} must declare {field}=true"
                        )

                proof.update(hash_expectations)
                proof.update(declared_token_counts)
                proof["context"] = context
                proof["validated"] = True
                normalized_proofs.append(proof)

    normalized = copy.deepcopy(draft)
    normalized.update(expected_manifest_metadata)
    normalized["schema_version"] = NEUTRAL_MANIFEST_SCHEMA_VERSION
    normalized["input_artifacts"] = input_artifacts
    normalized["replacements"] = normalized_proofs
    normalized["audit"] = {
        "status": "validated",
        "artifact_identity_universe_complete": True,
        "row_coverage_complete": True,
        "row_hashes_validated": True,
        "replacement_prefix_suffix_validated": True,
        "indicator_label_delimiters_validated": True,
        "block_and_full_prompt_token_lengths_validated": True,
        "distinct_neutral_prompts_per_indicator": True,
        "input_artifact_count": len(input_artifacts),
        "replacement_count": len(normalized_proofs),
        "token_count_policy": "tokenizer.encode(add_special_tokens=False)",
    }
    return normalized


__all__ = [
    "NEUTRAL_MANIFEST_SCHEMA_VERSION",
    "validate_neutral_intervention_manifest",
]
