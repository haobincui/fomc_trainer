"""Project indicator-analysis completions into canonical Minutes evidence.

The projection is deliberately deterministic.  It accepts only a well-formed
DeepSeek-style completion with exactly one ``</think>`` delimiter and writes
only the final answer after that delimiter.  It never truncates, summarizes,
or calls a generative model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from open_r1.provenance import fingerprint_artifact_path, sha256_file, sha256_text
from open_r1.structured_response import (
    DEEPSEEK_THINK_COMPLETION_FORMAT,
    parse_structured_response,
)


PROJECTION_SCHEMA_VERSION = "indicator-analysis-projection-v1"
PROJECTION_ROW_SCHEMA_VERSION = "indicator-analysis-projection-row-v1"
PROJECTION_POLICY = "deepseek-final-answer-after-think-v1"
SOURCE_FIELD = "generated"
OUTPUT_FIELD = "minutes_analysis"
THINK_DELIMITER = "</think>"


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ...


def _json_text(payload: Any) -> str:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _canonical_row_text(row: Mapping[str, Any]) -> str:
    return json.dumps(
        row,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_row_sha256(row: Mapping[str, Any]) -> str:
    return sha256_text(_canonical_row_text(row))


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_canonical_row_text(row) + "\n" for row in rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Indicator-analysis JSONL does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid indicator-analysis JSON in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Indicator-analysis row {line_number} must be a JSON object"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Indicator-analysis JSONL is empty: {path}")
    return rows


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def extract_minutes_analysis(text: object) -> tuple[str, str]:
    """Return non-empty reasoning and final answer from one strict completion."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Indicator-analysis completion must be a non-empty string")
    delimiter_count = text.count(THINK_DELIMITER)
    if delimiter_count != 1:
        raise ValueError(
            "Canonical Minutes projection requires exactly one "
            f"{THINK_DELIMITER!r} delimiter, found {delimiter_count}"
        )
    parsed = parse_structured_response(text)
    if (
        parsed.format_name != DEEPSEEK_THINK_COMPLETION_FORMAT
        or not parsed.is_well_formed
        or not parsed.reasoning.strip()
        or not parsed.answer.strip()
    ):
        raise ValueError(
            "Canonical Minutes projection requires a well-formed DeepSeek "
            "reasoning completion with non-empty reasoning and final answer"
        )
    return parsed.reasoning.strip(), parsed.answer.strip()


def _write_exclusive(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(
            f"Projection artifacts are immutable; choose a new output directory: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def project_indicator_analysis(
    *,
    input_jsonl: str | Path,
    analysis_manifest: str | Path,
    tokenizer: TokenizerLike,
    tokenizer_artifact: Mapping[str, Any],
    output_dir: str | Path,
    source_field: str = SOURCE_FIELD,
    output_field: str = OUTPUT_FIELD,
) -> dict[str, Any]:
    """Create a hash-bound final-answer projection and its manifest."""

    if source_field != SOURCE_FIELD or output_field != OUTPUT_FIELD:
        raise ValueError(
            "Canonical projection fields are frozen as "
            f"{SOURCE_FIELD!r} -> {OUTPUT_FIELD!r}"
        )
    source_path = Path(input_jsonl).expanduser().resolve()
    source_manifest_path = Path(analysis_manifest).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    output_path = destination / "minutes_analysis.jsonl"
    manifest_path = destination / "analysis_projection_manifest.json"
    if destination.exists():
        raise FileExistsError(
            "Projection output directory already exists; use a new run-scoped "
            f"directory: {destination}"
        )

    tokenizer_digest = str(tokenizer_artifact.get("sha256") or "").strip().lower()
    if (
        len(tokenizer_digest) != 64
        or any(character not in "0123456789abcdef" for character in tokenizer_digest)
    ):
        raise ValueError("tokenizer_artifact.sha256 must be a lowercase SHA256")

    source_manifest = _read_json_object(
        source_manifest_path,
        label="raw analysis manifest",
    )
    source_output = source_manifest.get("output")
    if not isinstance(source_output, Mapping):
        raise ValueError("Raw analysis manifest lacks output metadata")
    declared_output_path = Path(
        str(source_output.get("path") or "")
    ).expanduser().resolve()
    observed_source_sha256 = sha256_file(source_path)
    if (
        declared_output_path != source_path
        or source_output.get("sha256") != observed_source_sha256
    ):
        raise ValueError(
            "Raw analysis manifest output path/hash does not match --input"
        )

    raw_rows = _read_jsonl(source_path)
    projected_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    observed_sample_ids: set[str] = set()
    observed_keys: set[tuple[str, str]] = set()

    for position, raw in enumerate(raw_rows):
        sample_id = str(raw.get("sample_id") or "").strip()
        meeting_date = str(raw.get("meeting_date") or "").strip()
        indicator = str(raw.get("indicator") or "").strip()
        if not sample_id or not meeting_date or not indicator:
            raise ValueError(
                f"Indicator-analysis row {position + 1} lacks sample_id, "
                "meeting_date, or indicator"
            )
        key = (meeting_date, indicator)
        if sample_id in observed_sample_ids or key in observed_keys:
            raise ValueError(
                f"Duplicate projection identity at row {position + 1}: "
                f"sample_id={sample_id!r}, key={key!r}"
            )
        observed_sample_ids.add(sample_id)
        observed_keys.add(key)

        source_text = raw.get(source_field)
        source_text_sha256 = sha256_text(source_text) if isinstance(source_text, str) else ""
        declared_source_digest = raw.get(f"{source_field}_sha256")
        if declared_source_digest != source_text_sha256:
            raise ValueError(
                f"Indicator-analysis row {position + 1} has an invalid "
                f"{source_field}_sha256"
            )
        reasoning, answer = extract_minutes_analysis(source_text)
        answer_token_ids = tokenizer.encode(answer, add_special_tokens=False)
        if not isinstance(answer_token_ids, (list, tuple)) or not answer_token_ids:
            raise ValueError(
                f"Projected final answer has no tokenizer tokens for {sample_id!r}"
            )

        source_row_sha256 = canonical_row_sha256(raw)
        reasoning_sha256 = sha256_text(reasoning)
        answer_sha256 = sha256_text(answer)
        projected = {
            "schema_version": PROJECTION_ROW_SCHEMA_VERSION,
            "projection_position": position,
            "sample_id": sample_id,
            "meeting_date": meeting_date,
            "indicator": indicator,
            "source_field": source_field,
            "source_row_sha256": source_row_sha256,
            "source_text_sha256": source_text_sha256,
            output_field: answer,
            f"{output_field}_sha256": answer_sha256,
            f"{output_field}_token_count_no_special_tokens": len(answer_token_ids),
        }
        projected_rows.append(projected)
        inventory_rows.append(
            {
                "projection_position": position,
                "sample_id": sample_id,
                "meeting_date": meeting_date,
                "indicator": indicator,
                "source_row_sha256": source_row_sha256,
                "source_text_sha256": source_text_sha256,
                "reasoning_sha256": reasoning_sha256,
                f"{output_field}_sha256": answer_sha256,
                f"{output_field}_token_count_no_special_tokens": len(
                    answer_token_ids
                ),
            }
        )

    output_text = _jsonl_text(projected_rows)
    output_sha256 = sha256_text(output_text)
    token_counts = [
        int(row[f"{output_field}_token_count_no_special_tokens"])
        for row in inventory_rows
    ]
    manifest = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "status": "complete",
        "generation_performed": False,
        "truncation_performed": False,
        "summarization_performed": False,
        "projection": {
            "policy": PROJECTION_POLICY,
            "parser": (
                "open_r1.structured_response.parse_structured_response"
            ),
            "required_format": DEEPSEEK_THINK_COMPLETION_FORMAT,
            "delimiter": THINK_DELIMITER,
            "required_delimiter_count": 1,
            "allow_plain_text_fallback": False,
            "require_nonempty_reasoning": True,
            "require_nonempty_final_answer": True,
            "source_field": source_field,
            "output_field": output_field,
        },
        "input": {
            "path": str(source_path),
            "sha256": observed_source_sha256,
            "row_count": len(raw_rows),
        },
        "input_analysis_manifest": {
            "path": str(source_manifest_path),
            "sha256": sha256_file(source_manifest_path),
            "schema_version": source_manifest.get("schema_version"),
        },
        "output": {
            "path": str(output_path),
            "sha256": output_sha256,
            "row_count": len(projected_rows),
        },
        "tokenizer_artifact": dict(tokenizer_artifact),
        "token_count_policy": "tokenizer.encode(add_special_tokens=False)",
        "inventory": {
            "row_count": len(inventory_rows),
            "meeting_count": len({row["meeting_date"] for row in inventory_rows}),
            "indicator_count": len({row["indicator"] for row in inventory_rows}),
            "minimum_final_answer_tokens": min(token_counts),
            "maximum_final_answer_tokens": max(token_counts),
            "rows": inventory_rows,
        },
    }
    _write_exclusive(output_path, output_text)
    _write_exclusive(manifest_path, _json_text(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically extract final indicator-analysis answers for "
            "canonical Minutes prompt construction."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-field", default=SOURCE_FIELD)
    parser.add_argument("--output-field", default=OUTPUT_FIELD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer must be a frozen local artifact: {tokenizer_path}"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    manifest = project_indicator_analysis(
        input_jsonl=args.input,
        analysis_manifest=args.analysis_manifest,
        tokenizer=tokenizer,
        tokenizer_artifact=fingerprint_artifact_path(tokenizer_path),
        output_dir=args.output_dir,
        source_field=args.source_field,
        output_field=args.output_field,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "schema_version": manifest["schema_version"],
                "row_count": manifest["inventory"]["row_count"],
                "output": manifest["output"]["path"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
