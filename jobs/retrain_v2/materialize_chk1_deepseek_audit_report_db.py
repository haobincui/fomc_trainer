"""Materialize reviewed chk1 DeepSeek audit aggregates into a SQLite snapshot."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from jobs.retrain_v2.compress_chk1_reasoning import _sha256_file


class ReportDatabaseError(RuntimeError):
    """The reviewed analysis cannot be materialized safely."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportDatabaseError(f"unable to read {path}") from exc
    if not isinstance(value, dict):
        raise ReportDatabaseError(f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReportDatabaseError(f"{path}:{number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ReportDatabaseError(f"{path}:{number}: row is not an object")
        rows.append(value)
    return rows


def _by_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["group"]): row for row in rows}


def materialize(*, analysis_path: Path, row_audits_path: Path, output: Path) -> None:
    analysis_path = analysis_path.resolve()
    row_audits_path = row_audits_path.resolve()
    output = output.resolve()
    if output.exists():
        raise ReportDatabaseError(f"refusing to overwrite report database: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    analysis = _read_json(analysis_path)
    rows = _read_jsonl(row_audits_path)
    if analysis.get("integrity", {}).get("status") != "passed" or len(rows) != 237:
        raise ReportDatabaseError("analysis integrity or row population mismatch")
    temporary = output.parent / f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    connection = sqlite3.connect(temporary)
    try:
        outcome = analysis["outcomes"]
        violations = analysis["violations"]
        adjudication = analysis["two_stage_adjudication"]
        connection.execute(
            """CREATE TABLE headline (
                audited_rows INTEGER NOT NULL,
                provider_errors INTEGER NOT NULL,
                passed_rows INTEGER NOT NULL,
                pass_rate REAL NOT NULL,
                failed_rows INTEGER NOT NULL,
                failure_rate REAL NOT NULL,
                confirmed_violations INTEGER NOT NULL,
                false_positive_proposals INTEGER NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO headline VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                outcome["total"],
                analysis["integrity"]["provider_errors"],
                outcome["passed"],
                outcome["pass_rate"],
                outcome["failed"],
                outcome["failure_rate"],
                violations["confirmed_total"],
                adjudication["false_positive"],
            ),
        )
        connection.execute(
            """CREATE TABLE audit_summary (
                strict_source_only_gate TEXT NOT NULL,
                audited_rows INTEGER NOT NULL,
                full_candidate_rows INTEGER NOT NULL,
                unaudited_candidate_rows INTEGER NOT NULL,
                passed_rows INTEGER NOT NULL,
                pass_rate REAL NOT NULL,
                failed_rows INTEGER NOT NULL,
                failure_rate REAL NOT NULL,
                rows_with_major_violation INTEGER NOT NULL,
                rows_with_only_minor_violations INTEGER NOT NULL,
                confirmed_violations INTEGER NOT NULL,
                false_positive_proposals INTEGER NOT NULL,
                rows_rescued_from_false_failure INTEGER NOT NULL,
                causal_violations INTEGER NOT NULL,
                factual_violations INTEGER NOT NULL,
                numerical_violations INTEGER NOT NULL,
                think_violations INTEGER NOT NULL,
                answer_violations INTEGER NOT NULL,
                target_leakage INTEGER NOT NULL,
                provider_errors INTEGER NOT NULL,
                review_required INTEGER NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO audit_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                analysis["decision"]["strict_source_only_gate"],
                outcome["total"],
                analysis["scope"]["full_candidate_rows"],
                analysis["scope"]["unaudited_candidate_rows"],
                outcome["passed"],
                outcome["pass_rate"],
                outcome["failed"],
                outcome["failure_rate"],
                outcome["rows_with_major_violation"],
                outcome["rows_with_only_minor_violations"],
                violations["confirmed_total"],
                adjudication["false_positive"],
                adjudication["rows_rescued_from_false_failure"],
                violations["by_kind"].get("causal", 0),
                violations["by_kind"].get("factual", 0),
                violations["by_kind"].get("numerical", 0),
                violations["by_section"].get("think", 0),
                violations["by_section"].get("answer", 0),
                violations["target_leakage"],
                analysis["integrity"]["provider_errors"],
                analysis["integrity"]["review_required"],
            ),
        )

        severity_by_kind = Counter(
            (item["kind"], item["severity"])
            for row in rows
            for item in row["blocking_violations"]
        )
        connection.execute(
            """CREATE TABLE violation_kind (
                kind TEXT PRIMARY KEY,
                confirmed_violations INTEGER NOT NULL,
                rows_affected INTEGER NOT NULL,
                major_violations INTEGER NOT NULL,
                minor_violations INTEGER NOT NULL
            )"""
        )
        for kind in ("causal", "factual", "numerical"):
            connection.execute(
                "INSERT INTO violation_kind VALUES (?, ?, ?, ?, ?)",
                (
                    kind.capitalize(),
                    violations["by_kind"].get(kind, 0),
                    violations["rows_affected_by_kind"].get(kind, 0),
                    severity_by_kind[(kind, "major")],
                    severity_by_kind[(kind, "minor")],
                ),
            )

        connection.execute(
            """CREATE TABLE repair_group (
                repair_group TEXT PRIMARY KEY,
                total INTEGER NOT NULL,
                passed INTEGER NOT NULL,
                failed INTEGER NOT NULL,
                failure_rate REAL NOT NULL,
                ci_low REAL NOT NULL,
                ci_high REAL NOT NULL
            )"""
        )
        repair_labels = {
            "answer unchanged": "Answer unchanged",
            "recovered JSON-like answer": "JSON recovered",
            "stripped inline evidence IDs": "IDs stripped",
        }
        for row in analysis["failure_rates"]["by_answer_repair_group"]:
            connection.execute(
                "INSERT INTO repair_group VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    repair_labels[row["group"]],
                    row["total"],
                    row["passed"],
                    row["failed"],
                    row["failure_rate"],
                    row["failure_rate_wilson_95"][0],
                    row["failure_rate_wilson_95"][1],
                ),
            )

        connection.execute(
            """CREATE TABLE split_outcomes (
                split TEXT PRIMARY KEY,
                total INTEGER NOT NULL,
                passed INTEGER NOT NULL,
                failed INTEGER NOT NULL,
                failure_rate REAL NOT NULL
            )"""
        )
        split_rates = _by_group(analysis["failure_rates"]["by_split"])
        for split in ("train", "eval", "test"):
            row = split_rates[split]
            connection.execute(
                "INSERT INTO split_outcomes VALUES (?, ?, ?, ?, ?)",
                (split, row["total"], row["passed"], row["failed"], row["failure_rate"]),
            )

        connection.execute(
            """CREATE TABLE representative_failures (
                sample_id TEXT PRIMARY KEY,
                section TEXT NOT NULL,
                kind TEXT NOT NULL,
                severity TEXT NOT NULL,
                violation_count INTEGER NOT NULL,
                major_count INTEGER NOT NULL,
                candidate_quote TEXT NOT NULL,
                explanation TEXT NOT NULL
            )"""
        )
        for row in analysis["representative_confirmed_failures"]:
            connection.execute(
                "INSERT INTO representative_failures VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["sample_id"],
                    row["example_section"],
                    row["example_kind"],
                    row["example_severity"],
                    row["violation_count"],
                    row["major_count"],
                    row["example_quote"],
                    row["example_explanation"],
                ),
            )
        connection.execute(
            """CREATE TABLE source_files (
                role TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL
            )"""
        )
        connection.executemany(
            "INSERT INTO source_files VALUES (?, ?, ?)",
            [
                ("analysis", str(analysis_path), _sha256_file(analysis_path)),
                ("row_audits", str(row_audits_path), analysis["integrity"]["row_audits_sha256"]),
            ],
        )
        connection.commit()
        connection.close()
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            pass
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--row-audits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    materialize(
        analysis_path=args.analysis,
        row_audits_path=args.row_audits,
        output=args.output,
    )
    print(json.dumps({"status": "complete", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
