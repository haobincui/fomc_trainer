"""Read-only, reproducible validation for the clean chk1 SFT release.

The validator compares a repaired release with its sealed v1 source by row and
then exercises the exact prompt renderer/tokenizer path used by SFTTrainer.  It
never writes to either release.  Its JSON report intentionally contains only
aggregate measurements, hashes, and hash-addressed row locations; training
text is never copied into the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.retrain_v2.chk1.prompt_projection import ANALYSIS_SFT_SYSTEM_PROMPT
from jobs.retrain_v2.repair_chk1_sft_release import (
    BOUNDARY,
    SPLITS,
    _ANSWER_META_RE,
    _EVIDENCE_ID_RE,
    _MARKDOWN_RE,
    _has_strict_periodic_tail,
    is_json_like_answer,
)
from open_r1.trainer.sft_prompt_renderer import (
    SftPromptRenderError,
    render_sft_prompt,
    tokenize_sft_text,
)


SCHEMA_VERSION = "chk1-clean-sft-validator-v1"
DATA_CONTRACT_VERSION = "chk1-clean-sft-data-contract-v2"
TOKEN_CONTRACT_VERSION = "chk1-sft-single-bos-completion-mask-v1"
EXPECTED_SPLIT_COUNTS = {"train": 1354, "eval": 199, "test": 190}
EXPECTED_COLUMNS = {"prompt", "response", "provided_data"}

DEFAULT_SOURCE_RELEASE = Path(
    "dataset/processed/retrain_v2/chk1_reasoning_compressed_flash_max_v1_20260805"
)
DEFAULT_CLEAN_RELEASE = Path(
    "dataset/processed/retrain_v2/chk1_reasoning_compressed_flash_max_v2_clean_20260809"
)
DEFAULT_TOKENIZER = Path("models/DeepSeek-R1-Distill-Llama-8B")

_ANY_CONTROL_TAG_RE = re.compile(
    r"</?(?:think|answer)>|<[|\uff5c][^>]+[|\uff5c]>", flags=re.IGNORECASE
)
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
)


class CleanSftValidationError(RuntimeError):
    """The validator could not produce a trustworthy result."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile)))
    return ordered[index]


def _distribution(values: Sequence[int]) -> Mapping[str, int | None]:
    return {
        "min": min(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _dataset_dir(release: Path) -> Path:
    path = release / "analysis_sft"
    if not path.is_dir() or path.is_symlink():
        raise CleanSftValidationError(f"missing regular analysis_sft directory: {path}")
    return path


def _read_jsonl(path: Path) -> list[tuple[Mapping[str, Any], str]]:
    if not path.is_file() or path.is_symlink():
        raise CleanSftValidationError(f"missing regular JSONL file: {path}")
    rows: list[tuple[Mapping[str, Any], str]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                line = raw_line.rstrip("\n")
                if not line or line.endswith("\r"):
                    raise CleanSftValidationError(
                        f"invalid JSONL framing: {path}:{line_number}"
                    )
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CleanSftValidationError(
                        f"invalid JSONL row: {path}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise CleanSftValidationError(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                rows.append((row, _sha256_text(line)))
    except (OSError, UnicodeError) as exc:
        raise CleanSftValidationError(f"cannot read JSONL file: {path}") from exc
    return rows


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment contract
        raise CleanSftValidationError("transformers is required") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - tokenizer loaders vary
        raise CleanSftValidationError(f"cannot load local tokenizer: {path}") from exc
    if getattr(tokenizer, "bos_token_id", None) is None:
        raise CleanSftValidationError("tokenizer lacks BOS")
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise CleanSftValidationError("tokenizer lacks EOS")
    if not isinstance(getattr(tokenizer, "eos_token", None), str):
        raise CleanSftValidationError("tokenizer lacks a literal EOS token")
    return tokenizer


def _encode_without_special_tokens(tokenizer: Any, value: str) -> list[int]:
    try:
        token_ids = tokenizer.encode(value, add_special_tokens=False)
    except Exception as exc:  # noqa: BLE001 - tokenizer implementations vary
        raise CleanSftValidationError("tokenizer.encode failed") from exc
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise CleanSftValidationError("tokenizer.encode returned invalid IDs")
    return token_ids


def _input_descriptor(release: Path) -> Mapping[str, Any]:
    dataset = _dataset_dir(release)
    splits: dict[str, Mapping[str, Any]] = {}
    for split in SPLITS:
        path = dataset / f"{split}.jsonl"
        if not path.is_file() or path.is_symlink():
            raise CleanSftValidationError(f"missing regular split file: {path}")
        splits[split] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    manifest_files = []
    for name in ("candidate_manifest.json", "release_manifest.json"):
        path = release / name
        if path.is_file() and not path.is_symlink():
            manifest_files.append({"name": name, "sha256": _sha256_file(path)})
    return {
        "release_id": release.name,
        "splits": splits,
        "manifests": manifest_files,
    }


def _tokenizer_descriptor(path: Path) -> Mapping[str, Any]:
    files = []
    for name in _TOKENIZER_FILES:
        candidate = path / name
        if candidate.is_file() and not candidate.is_symlink():
            files.append(
                {
                    "name": name,
                    "bytes": candidate.stat().st_size,
                    "sha256": _sha256_file(candidate),
                }
            )
    if not files:
        raise CleanSftValidationError(f"no tokenizer assets found: {path}")
    return {
        "model_id": path.name,
        "files": files,
        "binding_sha256": _sha256_text(_canonical_json(files)),
        "tokenizer_bundle_sha256": _sha256_text(_canonical_json(files)),
    }


def _load_sample_identity(
    clean_release: Path,
) -> tuple[dict[tuple[str, int], Mapping[str, Any]], Mapping[str, Any]]:
    path = clean_release / "audits" / "repair_manifest.jsonl"
    rows = _read_jsonl(path)
    identities: dict[tuple[str, int], Mapping[str, Any]] = {}
    sample_ids: set[str] = set()
    for manifest_line, (row, _line_hash) in enumerate(rows, 1):
        split = row.get("split")
        line_number = row.get("source_line_number")
        sample_id = row.get("sample_id")
        if split not in SPLITS or not isinstance(line_number, int) or line_number <= 0:
            raise CleanSftValidationError(
                f"invalid repair-manifest coordinate at line {manifest_line}"
            )
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise CleanSftValidationError(
                f"invalid/duplicate repair-manifest sample_id at line {manifest_line}"
            )
        key = (split, line_number)
        if key in identities:
            raise CleanSftValidationError(
                f"duplicate repair-manifest coordinate at line {manifest_line}"
            )
        required_hashes = {}
        for name in ("prompt_sha256", "provided_data_sha256", "new_response_sha256"):
            value = row.get(name)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                raise CleanSftValidationError(
                    f"invalid repair-manifest hash at line {manifest_line}"
                )
            required_hashes[name] = value
        identities[key] = {"sample_id": sample_id, **required_hashes}
        sample_ids.add(sample_id)
    return identities, {
        "sha256": _sha256_file(path),
        "rows": len(rows),
        "sample_ids_sha256": _sha256_text(_canonical_json(sorted(sample_ids))),
    }


def _location(
    *, sample_id: str, split: str, line_number: int, row_sha256: str
) -> Mapping[str, Any]:
    return {
        "sample_id": sample_id,
        "split": split,
        "line_number": line_number,
        "row_sha256": row_sha256,
    }


def validate_clean_sft(
    *,
    source_release: Path,
    clean_release: Path,
    tokenizer_path: Path,
    max_length: int = 4608,
    expected_split_counts: Mapping[str, int] | None = None,
    tokenizer: Any | None = None,
) -> Mapping[str, Any]:
    """Validate one clean release and return a text-free deterministic report."""

    source_release = source_release.resolve()
    clean_release = clean_release.resolve()
    tokenizer_path = tokenizer_path.resolve()
    if max_length <= 0:
        raise CleanSftValidationError("max_length must be positive")
    expected = dict(expected_split_counts or EXPECTED_SPLIT_COUNTS)
    if set(expected) != set(SPLITS) or any(
        not isinstance(value, int) or value < 0 for value in expected.values()
    ):
        raise CleanSftValidationError("invalid expected split counts")

    source_descriptor = _input_descriptor(source_release)
    clean_descriptor = _input_descriptor(clean_release)
    identity_by_location, identity_descriptor = _load_sample_identity(clean_release)
    if tokenizer is None:
        tokenizer = _load_tokenizer(tokenizer_path)
    tokenizer_descriptor = _tokenizer_descriptor(tokenizer_path)

    issues: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    split_counts: dict[str, int] = {}
    prompt_unchanged = 0
    provided_data_unchanged = 0
    order_unchanged = 0
    valid_rows = 0
    row_bindings: list[Mapping[str, Any]] = []
    duplicate_index: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    token_values: dict[str, list[int]] = {
        "prompt": [],
        "reasoning": [],
        "answer": [],
        "completion": [],
        "total": [],
    }

    def add_issue(
        code: str,
        location: Mapping[str, Any],
        *,
        field: str | None = None,
        observed: int | str | None = None,
    ) -> None:
        issue: dict[str, Any] = {"code": code, **location}
        if field is not None:
            issue["field"] = field
        if observed is not None:
            issue["observed"] = observed
        issues.append(issue)
        issue_counts[code] += 1

    total_expected_rows = sum(expected.values())
    for split in SPLITS:
        source_path = _dataset_dir(source_release) / f"{split}.jsonl"
        clean_path = _dataset_dir(clean_release) / f"{split}.jsonl"
        source_rows = _read_jsonl(source_path)
        clean_rows = _read_jsonl(clean_path)
        split_counts[split] = len(clean_rows)
        split_location = _location(
            sample_id=f"__split__:{split}",
            split=split,
            line_number=0,
            row_sha256=_sha256_file(clean_path),
        )
        if len(clean_rows) != expected[split]:
            add_issue(
                "split_count_mismatch",
                split_location,
                observed=len(clean_rows),
            )
        if len(source_rows) != len(clean_rows):
            add_issue(
                "source_clean_row_count_mismatch",
                split_location,
                observed=len(source_rows),
            )

        common_rows = min(len(source_rows), len(clean_rows))
        for line_number in range(1, common_rows + 1):
            source_raw, _source_line_hash = source_rows[line_number - 1]
            clean_raw, clean_line_hash = clean_rows[line_number - 1]
            identity = identity_by_location.get((split, line_number))
            sample_id = (
                str(identity["sample_id"])
                if identity is not None
                else f"__missing__:{split}:{line_number}"
            )
            location = _location(
                sample_id=sample_id,
                split=split,
                line_number=line_number,
                row_sha256=clean_line_hash,
            )
            if identity is None:
                add_issue("repair_manifest_identity_missing", location)

            source_schema_ok = set(source_raw) == EXPECTED_COLUMNS and all(
                isinstance(source_raw.get(name), str) for name in EXPECTED_COLUMNS
            )
            if not source_schema_ok:
                raise CleanSftValidationError(
                    f"source row schema invalid: {split}:{line_number}"
                )
            clean_schema_ok = set(clean_raw) == EXPECTED_COLUMNS and all(
                isinstance(clean_raw.get(name), str) and bool(clean_raw[name])
                for name in EXPECTED_COLUMNS
            )
            if not clean_schema_ok:
                add_issue("clean_row_schema_or_empty_field", location)
                continue

            row = {name: str(clean_raw[name]) for name in EXPECTED_COLUMNS}
            source = {name: str(source_raw[name]) for name in EXPECTED_COLUMNS}
            prompt_sha = _sha256_text(row["prompt"])
            provided_sha = _sha256_text(row["provided_data"])
            response_sha = _sha256_text(row["response"])
            row_bindings.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "line_number": line_number,
                    "prompt_sha256": prompt_sha,
                    "provided_data_sha256": provided_sha,
                    "response_sha256": response_sha,
                }
            )
            if identity is not None:
                for manifest_name, observed_sha in (
                    ("prompt_sha256", prompt_sha),
                    ("provided_data_sha256", provided_sha),
                    ("new_response_sha256", response_sha),
                ):
                    if identity[manifest_name] != observed_sha:
                        add_issue(
                            "repair_manifest_hash_mismatch",
                            location,
                            field=manifest_name,
                        )

            prompt_same = prompt_sha == _sha256_text(source["prompt"])
            provided_same = provided_sha == _sha256_text(source["provided_data"])
            prompt_unchanged += int(prompt_same)
            provided_data_unchanged += int(provided_same)
            order_unchanged += int(prompt_same and provided_same)
            if not prompt_same:
                add_issue("source_prompt_hash_mismatch", location, field="prompt")
            if not provided_same:
                add_issue(
                    "source_provided_data_hash_mismatch",
                    location,
                    field="provided_data",
                )

            boundary_count = row["response"].count(BOUNDARY)
            closing_count = len(re.findall(r"</think>", row["response"], re.I))
            if boundary_count != 1 or closing_count != 1:
                add_issue(
                    "native_boundary_count",
                    location,
                    field="response",
                    observed=boundary_count,
                )
                continue
            reasoning, answer = row["response"].split(BOUNDARY, 1)
            if reasoning != reasoning.strip() or answer != answer.strip():
                add_issue("target_surrounding_whitespace", location, field="response")
            if re.search(r"<think>", row["response"], re.I):
                add_issue("opening_think_tag", location, field="response")
            response_without_native_boundary = row["response"].replace(BOUNDARY, "", 1)
            if _ANY_CONTROL_TAG_RE.search(response_without_native_boundary):
                add_issue("control_tag_contamination", location, field="response")

            normalized_reasoning = _normalize_whitespace(reasoning)
            normalized_answer = _normalize_whitespace(answer)
            for field, value in (
                ("reasoning", reasoning),
                ("answer", answer),
            ):
                if is_json_like_answer(value):
                    add_issue("json_like_target", location, field=field)
                if _EVIDENCE_ID_RE.search(value):
                    add_issue("evidence_id_contamination", location, field=field)
                if _ANSWER_META_RE.search(value):
                    add_issue("meta_phrase_contamination", location, field=field)
                if _MARKDOWN_RE.search(value):
                    add_issue("markdown_contamination", location, field=field)
            if (
                normalized_answer
                and normalized_answer.casefold() in normalized_reasoning.casefold()
            ):
                add_issue("answer_reproduced_in_reasoning", location, field="reasoning")

            reasoning_ids = _encode_without_special_tokens(tokenizer, reasoning)
            answer_ids = _encode_without_special_tokens(tokenizer, answer)
            token_values["reasoning"].append(len(reasoning_ids))
            token_values["answer"].append(len(answer_ids))
            if not 512 <= len(reasoning_ids) <= 2400:
                add_issue(
                    "reasoning_token_range",
                    location,
                    field="reasoning",
                    observed=len(reasoning_ids),
                )
            if not 16 <= len(answer_ids) <= 512:
                add_issue(
                    "answer_token_range",
                    location,
                    field="answer",
                    observed=len(answer_ids),
                )
            response_ids = _encode_without_special_tokens(tokenizer, row["response"])
            if _has_strict_periodic_tail(response_ids):
                add_issue("strict_periodic_tail", location, field="response")

            try:
                rendered_prompt = render_sft_prompt(
                    tokenizer,
                    [
                        {"role": "system", "content": ANALYSIS_SFT_SYSTEM_PROMPT},
                        {"role": "user", "content": row["prompt"]},
                    ],
                )
                completion = row["response"]
                eos_token = tokenizer.eos_token
                if not completion.endswith(eos_token):
                    completion += eos_token
                prompt_ids = tokenize_sft_text(tokenizer, rendered_prompt)
                full_ids = tokenize_sft_text(tokenizer, rendered_prompt + completion)
            except (SftPromptRenderError, CleanSftValidationError):
                add_issue("sft_tokenization_contract", location)
                continue

            prompt_is_prefix = full_ids[: len(prompt_ids)] == prompt_ids
            if not prompt_is_prefix:
                add_issue("prompt_prefix_mismatch", location)
            completion_ids = full_ids[len(prompt_ids) :] if prompt_is_prefix else []
            labels = [-100] * len(prompt_ids) + completion_ids
            if (
                len(labels) != len(full_ids)
                or any(value != -100 for value in labels[: len(prompt_ids)])
                or not completion_ids
                or labels[-1] == -100
            ):
                add_issue("completion_mask_contract", location)

            bos_token_id = int(tokenizer.bos_token_id)
            eos_token_id = int(tokenizer.eos_token_id)
            if (
                not full_ids
                or full_ids[0] != bos_token_id
                or full_ids.count(bos_token_id) != 1
            ):
                add_issue(
                    "single_bos_contract",
                    location,
                    observed=full_ids.count(bos_token_id),
                )
            if (
                not full_ids
                or full_ids[-1] != eos_token_id
                or full_ids.count(eos_token_id) != 1
            ):
                add_issue(
                    "final_eos_contract",
                    location,
                    observed=full_ids.count(eos_token_id),
                )
            token_values["prompt"].append(len(prompt_ids))
            token_values["completion"].append(len(completion_ids))
            token_values["total"].append(len(full_ids))
            if len(full_ids) > max_length:
                add_issue(
                    "total_length_overflow",
                    location,
                    observed=len(full_ids),
                )

            for field, value in (
                ("prompt", row["prompt"]),
                ("provided_data", row["provided_data"]),
                ("reasoning", reasoning),
                ("answer", answer),
                ("response", row["response"]),
            ):
                for mode, normalized in (
                    ("exact", value),
                    ("whitespace_normalized", _normalize_whitespace(value)),
                ):
                    duplicate_index[(field, mode, _sha256_text(normalized))].append(
                        location
                    )
            valid_rows += 1

    manifest_coordinates = set(identity_by_location)
    expected_coordinates = {
        (split, line_number)
        for split in SPLITS
        for line_number in range(1, split_counts.get(split, 0) + 1)
    }
    for split, line_number in sorted(manifest_coordinates - expected_coordinates):
        identity = identity_by_location[(split, line_number)]
        add_issue(
            "repair_manifest_extra_identity",
            _location(
                sample_id=str(identity["sample_id"]),
                split=split,
                line_number=line_number,
                row_sha256=str(identity["new_response_sha256"]),
            ),
        )

    duplicate_group_counts: Counter[str] = Counter()
    for (field, mode, value_sha), locations in sorted(duplicate_index.items()):
        if len({str(location["split"]) for location in locations}) <= 1:
            continue
        code = "cross_split_duplicate"
        duplicate_group_counts[f"{mode}:{field}"] += 1
        issues.append(
            {
                "code": code,
                "field": field,
                "mode": mode,
                "value_sha256": value_sha,
                "locations": locations,
            }
        )
        issue_counts[code] += 1

    issues.sort(
        key=lambda item: (
            str(item.get("code", "")),
            str(item.get("split", "")),
            int(item.get("line_number", 0)),
            str(item.get("field", "")),
            str(item.get("value_sha256", "")),
        )
    )
    observed_rows = sum(split_counts.values())
    gates = {
        "split_counts_and_order": (
            split_counts == expected and order_unchanged == total_expected_rows
        ),
        "prompt_hashes_unchanged": prompt_unchanged == total_expected_rows,
        "provided_data_hashes_unchanged": provided_data_unchanged
        == total_expected_rows,
        "schema_and_target_contract": not any(
            code in issue_counts
            for code in (
                "clean_row_schema_or_empty_field",
                "native_boundary_count",
                "target_surrounding_whitespace",
                "opening_think_tag",
                "control_tag_contamination",
                "json_like_target",
                "evidence_id_contamination",
                "meta_phrase_contamination",
                "markdown_contamination",
                "answer_reproduced_in_reasoning",
                "reasoning_token_range",
                "answer_token_range",
            )
        ),
        "cross_split_duplicates": issue_counts["cross_split_duplicate"] == 0,
        "strict_periodic_tails": issue_counts["strict_periodic_tail"] == 0,
        "single_bos_final_eos_and_mask": not any(
            code in issue_counts
            for code in (
                "sft_tokenization_contract",
                "prompt_prefix_mismatch",
                "completion_mask_contract",
                "single_bos_contract",
                "final_eos_contract",
            )
        ),
        "no_truncation_at_max_length": issue_counts["total_length_overflow"] == 0,
        "repair_manifest_binding": not any(
            code in issue_counts
            for code in (
                "repair_manifest_identity_missing",
                "repair_manifest_hash_mismatch",
                "repair_manifest_extra_identity",
            )
        ),
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "validator": {
            "version": SCHEMA_VERSION,
            "source_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "contracts": {
            "data": DATA_CONTRACT_VERSION,
            "tokenization": TOKEN_CONTRACT_VERSION,
        },
        "status": "passed" if not issues and all(gates.values()) else "failed",
        "configuration": {
            "expected_split_counts": expected,
            "max_length": max_length,
            "reasoning_token_range": [512, 2400],
            "answer_token_range": [16, 512],
        },
        "inputs": {
            "source_release": source_descriptor,
            "clean_release": clean_descriptor,
            "repair_manifest": identity_descriptor,
            "tokenizer": tokenizer_descriptor,
        },
        "counts": {
            "expected_rows": total_expected_rows,
            "observed_rows": observed_rows,
            "valid_rows": valid_rows,
            "split_counts": split_counts,
            "prompt_hashes_unchanged": prompt_unchanged,
            "provided_data_hashes_unchanged": provided_data_unchanged,
            "order_unchanged": order_unchanged,
            "truncated_rows": issue_counts["total_length_overflow"],
            "issue_rows_or_groups": len(issues),
        },
        "content_binding_sha256": _sha256_text(_canonical_json(row_bindings)),
        "token_statistics": {
            "max_total_tokens": max(token_values["total"])
            if token_values["total"]
            else None,
            "distributions": {
                name: _distribution(values) for name, values in token_values.items()
            },
        },
        "cross_split_duplicate_group_counts": dict(
            sorted(duplicate_group_counts.items())
        ),
        "quality_gates": gates,
        "issue_counts": dict(sorted(issue_counts.items())),
        "issues": issues,
    }
    report["validation_sha256"] = _sha256_text(_canonical_json(report))
    return report


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, default=DEFAULT_SOURCE_RELEASE)
    parser.add_argument("--clean-release", type=Path, default=DEFAULT_CLEAN_RELEASE)
    parser.add_argument("--model-tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-length", type=int, default=4608)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = validate_clean_sft(
            source_release=args.source_release,
            clean_release=args.clean_release,
            tokenizer_path=args.model_tokenizer,
            max_length=args.max_length,
        )
    except CleanSftValidationError as exc:
        # A fatal report remains text-free with respect to training data.  The
        # exception digest makes repeated failures comparable without copying
        # provider/dataset contents into an artifact.
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "fatal_error": {
                "type": type(exc).__name__,
                "sha256": _sha256_text(str(exc)),
            },
        }
        report["validation_sha256"] = _sha256_text(_canonical_json(report))
    _write_json_atomic(args.output, report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())
