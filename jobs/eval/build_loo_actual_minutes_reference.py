"""Build a frozen 39-row Actual-Minutes reference view for scoped LOO runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from open_r1.provenance import sha256_file
from open_r1.validator.loo_experiment_scope import (
    load_and_validate_experiment_scope,
)


REFERENCE_SCHEMA_VERSION = "loo-actual-minutes-reference-view-v1"
SECTION_NAME_ALIASES = {
    "Participants' Views on Current Economic Conditions and the Economic Outlook": (
        "Participants' Views on Current Conditions and the Economic Outlook"
    ),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Actual-Minutes source is empty: {path}")
    return rows


def _jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_population_manifest(
    path: Path,
    *,
    scope: dict[str, Any],
    population_id: str,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Population manifest does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid population manifest JSON {resolved}: {exc}") from exc
    scoped = scope["populations"].get(population_id)
    if not isinstance(payload, dict) or not isinstance(scoped, dict):
        raise ValueError(f"Invalid population manifest for {population_id!r}")
    expected = {
        "schema_version": "loo-population-v1",
        "population_id": population_id,
        "phase": scoped["phase"],
        "split_label": scoped["split_label"],
        "meeting_dates": scoped["meeting_dates"],
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Population manifest differs from scoped population {population_id!r}: "
            f"{mismatches}"
        )
    result = dict(payload)
    result["_path"] = str(resolved)
    result["_sha256"] = sha256_file(resolved)
    return result


def build_actual_minutes_reference(
    *,
    source_file: Path,
    population_manifest_file: Path,
    scope_manifest_file: Path,
    population_id: str,
    output_file: Path,
    manifest_file: Path | None = None,
    expected_source_sha256: str | None = None,
    source_text_field: str = "reference",
    output_text_field: str = "response",
) -> dict[str, Any]:
    """Create a deterministic meeting-section reference and provenance manifest."""

    source_path = source_file.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Actual-Minutes source does not exist: {source_path}")
    source_sha256 = sha256_file(source_path)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise ValueError(
            "Actual-Minutes source SHA-256 differs from --expected-source-sha256: "
            f"expected={expected_source_sha256}, observed={source_sha256}"
        )

    scope_path = scope_manifest_file.expanduser().resolve()
    scope = load_and_validate_experiment_scope(scope_path)
    scope["_path"] = str(scope_path)
    scope["_sha256"] = sha256_file(scope_path)
    population = _load_population_manifest(
        population_manifest_file,
        scope=scope,
        population_id=population_id,
    )
    meeting_dates = list(population["meeting_dates"])
    sections = list(scope["section_names"])
    expected_keys = {
        (meeting_date, section_name)
        for meeting_date in meeting_dates
        for section_name in sections
    }

    source_rows = _read_jsonl(source_path)
    selected: dict[tuple[str, str], dict[str, str]] = {}
    selected_source_sections: dict[tuple[str, str], str] = {}
    for line_number, row in enumerate(source_rows, 1):
        meeting_date = str(row.get("meeting_date") or "").strip()[:10]
        source_section = str(row.get("section_name") or "").strip()
        section_name = SECTION_NAME_ALIASES.get(source_section, source_section)
        key = (meeting_date, section_name)
        if key not in expected_keys:
            continue
        text_value = row.get(source_text_field)
        if not isinstance(text_value, str) or not text_value.strip():
            raise ValueError(
                f"Actual-Minutes source has empty {source_text_field!r} for "
                f"{key!r} at line {line_number}"
            )
        if key in selected:
            raise ValueError(
                f"Actual-Minutes source maps multiple rows to scoped key {key!r}"
            )
        selected[key] = {
            "meeting_date": meeting_date,
            "section_name": section_name,
            output_text_field: text_value.strip(),
        }
        selected_source_sections[key] = source_section

    observed_keys = set(selected)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise ValueError(
            "Actual-Minutes reference does not cover the scoped 13x3 matrix: "
            f"missing={missing}, extra={extra}"
        )

    output_rows = [
        selected[(meeting_date, section_name)]
        for meeting_date in meeting_dates
        for section_name in sections
    ]
    if len(output_rows) != 39:
        raise AssertionError(f"Expected 39 Actual-Minutes rows, observed {len(output_rows)}")

    output_path = output_file.expanduser().resolve()
    output_text = _jsonl_text(output_rows)
    _atomic_write_text(output_path, output_text)
    output_sha256 = sha256_file(output_path)
    manifest_path = (
        manifest_file.expanduser().resolve()
        if manifest_file is not None
        else output_path.with_name(f"{output_path.stem}_manifest.json")
    )
    aliases_used = {
        source_section: canonical_section
        for (meeting_date, canonical_section), source_section in selected_source_sections.items()
        if source_section != canonical_section
    }
    manifest: dict[str, Any] = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "status": "complete",
        "experiment_id": scope["experiment_id"],
        "population_id": population_id,
        "phase": population["phase"],
        "row_count": len(output_rows),
        "section_name_aliases": dict(sorted(aliases_used.items())),
        "scope_manifest": {
            "path": scope["_path"],
            "sha256": scope["_sha256"],
        },
        "population_manifest": {
            "path": population["_path"],
            "sha256": population["_sha256"],
        },
        "source": {
            "path": str(source_path),
            "sha256": source_sha256,
            "row_count": len(source_rows),
            "text_field": source_text_field,
        },
        "output": {
            "path": str(output_path),
            "sha256": output_sha256,
            "row_count": len(output_rows),
            "key_fields": ["meeting_date", "section_name"],
            "text_field": output_text_field,
        },
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a frozen 39-row Actual-Minutes reference for one LOO population."
    )
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--scope-manifest", type=Path, required=True)
    parser.add_argument(
        "--population-id",
        choices=["pilot_eval_13", "formal_test_13"],
        required=True,
    )
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--manifest-file", type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--source-text-field", default="reference")
    parser.add_argument("--output-text-field", default="response")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_actual_minutes_reference(
        source_file=args.source_file,
        population_manifest_file=args.population_manifest,
        scope_manifest_file=args.scope_manifest,
        population_id=args.population_id,
        output_file=args.output_file,
        manifest_file=args.manifest_file,
        expected_source_sha256=args.expected_source_sha256,
        source_text_field=args.source_text_field,
        output_text_field=args.output_text_field,
    )
    print("Actual-Minutes reference complete")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
