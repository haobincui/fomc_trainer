"""Prepare exact-D-1 evidence for the 1993--2008 chk4 Decision supplement.

This module never calls DeepSeek.  It isolates the only safe fields from the
archived Decision workbook, freezes nine 11/13-meeting ALFRED populations,
materializes the existing sparse exact-vintage source pipeline, and converts
validated ledger rows into compact, date-free model inputs.

The model-facing files contain opaque sample IDs and target-neutral evidence.
Meeting dates and canonical policy labels remain in manifests only.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from jobs.generation.generate_chk4_sft_targets import (
    DEFAULT_FFR_HISTORY,
    DEFAULT_TOKENIZER,
    DEFAULT_WORKBOOK,
    canonical_json,
    load_and_validate_labels,
    sha256_file,
    sha256_text,
)
from jobs.main.fetch_loo_source_snapshots import HttpResult, load_source_registry
from jobs.retrain_v2.chk1.source_pipeline import (
    POPULATION_SCHEMA_VERSION,
    SOURCE_PLAN_SCHEMA_VERSION,
    materialize_source_plan,
    validate_source_handoff,
)
from open_r1.validator.loo_ledger import _load_registry, _load_roster


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk4/decision_supplement_1993_2008_v1"
)
DEFAULT_BASE_ROSTER = REPO_ROOT / "configs/main/leave_one_out_roster.json"
DEFAULT_BASE_REGISTRY = REPO_ROOT / "configs/main/loo_indicator_sources.json"

POPULATION = "decision_supplement_1993_2008_v1"
ROSTER_ID = "chapter2-historical-26-indicators-1993-2008-v1"
REGISTRY_ID = "chapter2-canonical-loo-keyless-d1-1993-2008-v1"
EXPECTED_CANDIDATES = 109
MIN_ADMITTED = EXPECTED_CANDIDATES
MIN_VALID_ATOMIC_TOPICS = 8
ADMISSION_PROFILE = "historical-sparse-three-category-min8-v2"
EXPECTED_ACTION_COUNTS = {
    "cut:25": 9,
    "cut:50": 8,
    "cut:75": 2,
    "hike:25": 20,
    "hike:50": 3,
    "hold:0": 67,
}
EXPECTED_POPULATION_SIZES = (13, 13, 13, 13, 13, 11, 11, 11, 11)
MAX_SERIES_PER_TOPIC = 2
CHANGE_LAGS = ((1, "previous"), (3, "short"), (6, "medium"), (12, "long"))
MAX_SUMMARY_INPUT_TOKENS = 12_288

CANDIDATE_SCHEMA = "chk4-decision-supplement-candidate-v1"
PREPARED_SCHEMA = "chk4-decision-supplement-summary-request-v1"
EVIDENCE_SCHEMA = "chk4-decision-supplement-evidence-v1"
ADMISSION_SCHEMA = "chk4-decision-supplement-admission-v1"
SOURCE_CONTRACT_SCHEMA = "chk4-decision-supplement-source-contract-v1"

PRICE_TOPICS = frozenset(
    {
        "Commodity-Prices",
        "Consumer-Price-Index-(CPI)",
        "Home-Prices",
        "Personal-Consumption-Expenditures-(PCE)",
    }
)
ACTIVITY_TOPICS = frozenset(
    {
        "Business-Investment",
        "Consumer-Confidence-Index",
        "GDP-Growth",
        "Government-Purchases",
        "Housing-Starts",
        "Industrial-Production",
        "Labour-Market",
        "Trade-Balance",
        "Unemployment-Rate",
    }
)
FINANCIAL_TOPICS = frozenset(
    {
        "Bank-Capital",
        "Bank-Credit-to-Private-Sector",
        "Corporate-Bond-Yields",
        "Equity-Market-Indices",
        "Exchange-Rate",
        "Federal-Funds-Rate",
        "Federal-Reserve-Balance-Sheet",
        "International-Equity-Markets",
        "Market-Volatility-(VIX)",
        "Money-Supply",
        "Mortgage-Rates",
        "Overnight-Rate",
        "Treasury-Yields",
    }
)
REQUIRED_CATEGORIES = frozenset(
    {"prices", "employment_activity", "financial_conditions"}
)

SUMMARY_SYSTEM_PROMPT = """\
You are a target-neutral pre-meeting macroeconomic evidence summarizer. Use
only the supplied point-in-time topic evidence. Reconcile the evidence across
inflation, employment and real activity, financial conditions, and the balance
of risks. Preserve every material number, direction, comparison, unit, and
expression of uncertainty needed for a later policy analysis.

Do not infer or mention the meeting identity. Do not use outside or remembered
historical information. Do not state, recommend, predict, or imply a policy
decision, vote, rate change, target range, or action actually taken. Do not
mention gold labels, Minutes, teacher targets, hidden fields, prompts, schemas,
or internal source IDs. Add no fact, cause, number, or date absent from the
evidence.

Return exactly one JSON object with the single key meeting_decision_brief. Its
value must be one coherent formal paragraph without headings, lists,
recommendations, citations, policy actions, or embedded JSON.
"""
SUMMARY_USER_PREFIX = (
    "Summarize the following target-neutral point-in-time topic evidence:\n\n"
)


class SupplementPreparationError(RuntimeError):
    """The supplement cannot be prepared without violating its contract."""


def _supplement_alfred_http_get(url: str, timeout_seconds: float) -> HttpResult:
    """Use ALFRED's original URL without the legacy custom User-Agent.

    ALFRED currently leaves requests carrying ``fomc-trainer-canonical-loo/1``
    open until the client read timeout, while the identical keyless URL with
    urllib's standard transport identity returns its deterministic status.
    This adapter changes no URL, query, vintage, response bytes, or PIT rule;
    it is scoped only to this supplement acquisition.
    """

    request = Request(url)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return HttpResult(
                status_code=int(response.status),
                headers={key.lower(): value for key, value in response.headers.items()},
                body=response.read(),
            )
    except HTTPError as exc:
        return HttpResult(
            status_code=int(exc.code),
            headers={key.lower(): value for key, value in exc.headers.items()},
            body=exc.read(),
        )


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    meeting_date: str
    direction: str
    magnitude_bp: int

    @property
    def gold(self) -> dict[str, Any]:
        return {"direction": self.direction, "magnitude_bp": self.magnitude_bp}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupplementPreparationError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SupplementPreparationError(f"JSON root is not an object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SupplementPreparationError(f"required JSONL is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SupplementPreparationError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(payload, dict):
                raise SupplementPreparationError(
                    f"JSONL row is not an object: {path}:{line_number}"
                )
            rows.append(payload)
    return rows


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SupplementPreparationError(f"immutable artifact drift: {path}")
        return
    _atomic_write(path, text)


def _write_json(path: Path, payload: Mapping[str, Any], *, immutable: bool = False) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (_write_immutable if immutable else _atomic_write)(path, text)


def _write_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], *, immutable: bool = False
) -> None:
    text = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    (_write_immutable if immutable else _atomic_write)(path, text)


def _opaque_sample_id(meeting_date: str) -> str:
    digest = sha256_text(f"chk4-decision-v1\0{POPULATION}\0{meeting_date}")[:24]
    return f"dec-{digest}"


def load_candidates(
    workbook: Path, ffr_history: Path
) -> tuple[list[Candidate], dict[str, Any]]:
    labels, audit = load_and_validate_labels(workbook, ffr_history)
    candidates = [
        Candidate(_opaque_sample_id(meeting), meeting, direction, magnitude)
        for meeting, (direction, magnitude) in sorted(labels.items())
        if "1993-01-01" <= meeting <= "2008-12-31"
    ]
    if len(candidates) != EXPECTED_CANDIDATES:
        raise SupplementPreparationError(
            f"supplement candidate count changed: {len(candidates)}"
        )
    counts = Counter(
        f"{candidate.direction}:{candidate.magnitude_bp}" for candidate in candidates
    )
    if dict(sorted(counts.items())) != EXPECTED_ACTION_COUNTS:
        raise SupplementPreparationError(
            f"supplement action distribution changed: {dict(counts)}"
        )
    if candidates[0].meeting_date != "1993-03-23" or candidates[-1].meeting_date != "2008-12-16":
        raise SupplementPreparationError("supplement date bounds changed")
    return candidates, {
        **audit,
        "population": POPULATION,
        "candidate_count": len(candidates),
        "date_min": candidates[0].meeting_date,
        "date_max": candidates[-1].meeting_date,
        "supplement_action_counts": dict(sorted(counts.items())),
    }


def _candidate_rows(candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": CANDIDATE_SCHEMA,
            "sample_id": candidate.sample_id,
            "meeting_date": candidate.meeting_date,
            "split": "train",
            "population": POPULATION,
            "population_role": "supplement",
            "gold": candidate.gold,
            "gold_sha256": sha256_text(canonical_json(candidate.gold)),
        }
        for candidate in candidates
    ]


def _derive_source_contracts(
    *, output_root: Path, base_roster: Path, base_registry: Path
) -> tuple[Path, Path, dict[str, Any]]:
    roster = deepcopy(_load_json(base_roster))
    indicators = roster.get("indicators")
    if not isinstance(indicators, list) or len(indicators) != 26:
        raise SupplementPreparationError("base roster is not the frozen 26-topic roster")
    roster["roster_id"] = ROSTER_ID
    roster["contexts"] = ["1993_2008_pre_meeting_d1"]
    roster["derived_from"] = {
        "path": str(base_roster.resolve()),
        "sha256": sha256_file(base_roster),
    }

    registry = deepcopy(_load_json(base_registry))
    if set(registry.get("indicators") or {}) != set(indicators):
        raise SupplementPreparationError("base registry and roster topics differ")
    policy = registry.get("policy")
    if not isinstance(policy, dict) or any(
        policy.get(key) != expected
        for key, expected in {
            "vintage_lag_calendar_days": 1,
            "same_meeting_day_data": "excluded",
            "unknown_availability": "fail_closed",
            "runtime_fallback": "forbidden",
        }.items()
    ):
        raise SupplementPreparationError("base registry is not the exact D-1 contract")
    registry["registry_id"] = REGISTRY_ID
    registry["roster_id"] = ROSTER_ID
    registry["derived_from"] = {
        "path": str(base_registry.resolve()),
        "sha256": sha256_file(base_registry),
    }

    roster_path = output_root / "sources/contracts/leave_one_out_roster_pre2009.json"
    registry_path = output_root / "sources/contracts/loo_indicator_sources_pre2009.json"
    _write_json(roster_path, roster, immutable=True)
    _write_json(registry_path, registry, immutable=True)
    # Replay both sides of the existing source interface during preflight so a
    # derived roster/registry mismatch is detected before any network request.
    load_source_registry(registry_path)
    roster_values = _load_roster(roster_path)
    _load_registry(registry_path, roster=roster_values)
    contract = {
        "schema_version": SOURCE_CONTRACT_SCHEMA,
        "population": POPULATION,
        "roster": {"path": str(roster_path), "sha256": sha256_file(roster_path)},
        "registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
        },
        "point_in_time": {
            "vintage": "meeting_date_minus_1_calendar_day",
            "current_vintage_fallback": "forbidden",
            "same_meeting_day_data": "excluded",
            "lookback_months": 24,
        },
        "structural_unavailability": {
            "topic": "Overnight-Rate",
            "series": "SOFR",
            "policy": "allow exact-vintage acquisition to seal unusable; never synthesize",
        },
        "base_roster_sha256": sha256_file(base_roster),
        "base_registry_sha256": sha256_file(base_registry),
    }
    contract["contract_sha256"] = sha256_text(canonical_json(contract))
    _write_json(output_root / "sources/contracts/source_contract.json", contract, immutable=True)
    return roster_path, registry_path, contract


def _source_plan_payload(
    candidates: Sequence[Candidate], destination: Path
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    dates = [candidate.meeting_date for candidate in candidates]
    populations: list[dict[str, Any]] = []
    population_files: list[tuple[Path, dict[str, Any]]] = []
    cursor = 0
    for index, size in enumerate(EXPECTED_POPULATION_SIZES, 1):
        chunk = dates[cursor : cursor + size]
        cursor += size
        population_id = f"chk4supp_train_{size}_{index:02d}"
        population_payload = {
            "schema_version": POPULATION_SCHEMA_VERSION,
            "population_id": population_id,
            "phase": "chk4_decision_supplement",
            "split_label": "train",
            "meeting_dates": chunk,
        }
        path = destination / "populations" / f"{population_id}.json"
        population_files.append((path, population_payload))
        rendered = json.dumps(
            population_payload, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        population_sha = sha256_text(rendered)
        populations.append(
            {
                "population_id": population_id,
                "split": "train",
                "meeting_dates": chunk,
                "path": str(path.resolve()),
                "sha256": population_sha,
                "existing": False,
            }
        )
    if cursor != len(dates):
        raise AssertionError("supplement partition closure failed")

    thirteen = [row for row in populations if len(row["meeting_dates"]) == 13]
    eleven = [row for row in populations if len(row["meeting_dates"]) == 11]
    batches: list[dict[str, Any]] = []
    for index in range(0, len(thirteen), 2):
        group = thirteen[index : index + 2]
        batches.append(
            {
                "batch_id": "snapshot_" + "__".join(row["population_id"] for row in group),
                "population_ids": [row["population_id"] for row in group],
                "expected_vintage_count": sum(len(row["meeting_dates"]) for row in group),
            }
        )
    for row in eleven:
        batches.append(
            {
                "batch_id": f"snapshot_{row['population_id']}",
                "population_ids": [row["population_id"]],
                "expected_vintage_count": 11,
            }
        )
    payload = {
        "schema_version": SOURCE_PLAN_SCHEMA_VERSION,
        "meeting_counts": {"train": len(dates), "eval": 0, "test": 0},
        "populations": populations,
        "snapshot_batches": batches,
    }
    return {**payload, "payload_sha256": sha256_text(canonical_json(payload))}, population_files


def _freeze_source_plan(candidates: Sequence[Candidate], output_root: Path) -> Path:
    destination = output_root / "sources/plan"
    plan, population_files = _source_plan_payload(candidates, destination)
    for path, payload in population_files:
        _write_json(path, payload, immutable=True)
        record = next(
            row for row in plan["populations"] if row["path"] == str(path.resolve())
        )
        if sha256_file(path) != record["sha256"]:
            raise SupplementPreparationError(f"population SHA mismatch: {path}")
    plan_path = destination / "source_plan.json"
    _write_json(plan_path, plan, immutable=True)
    return plan_path


def prepare_release(
    *,
    output_root: Path,
    workbook: Path,
    ffr_history: Path,
    base_roster: Path,
    base_registry: Path,
) -> dict[str, Any]:
    candidates, label_audit = load_candidates(workbook, ffr_history)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        output_root / "manifests/candidates.jsonl",
        _candidate_rows(candidates),
        immutable=True,
    )
    roster_path, registry_path, source_contract = _derive_source_contracts(
        output_root=output_root,
        base_roster=base_roster,
        base_registry=base_registry,
    )
    plan_path = _freeze_source_plan(candidates, output_root)
    summary = {
        "schema_version": "chk4-decision-supplement-preflight-v1",
        "status": "prepared",
        "api_requests": 0,
        "network_requests": 0,
        "label_audit": label_audit,
        "candidate_manifest_sha256": sha256_file(
            output_root / "manifests/candidates.jsonl"
        ),
        "source_contract_sha256": source_contract["contract_sha256"],
        "source_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "roster": {"path": str(roster_path), "sha256": sha256_file(roster_path)},
        "registry": {"path": str(registry_path), "sha256": sha256_file(registry_path)},
        "population_sizes": list(EXPECTED_POPULATION_SIZES),
    }
    _write_json(output_root / "reports/preflight.json", summary)
    _update_root_summary(output_root, "preflight", summary)
    return summary


def acquire_sources(
    *, output_root: Path, resume: bool, requests_per_second: float, max_workers: int
) -> dict[str, Any]:
    plan = output_root / "sources/plan/source_plan.json"
    roster = output_root / "sources/contracts/leave_one_out_roster_pre2009.json"
    registry = output_root / "sources/contracts/loo_indicator_sources_pre2009.json"
    for path in (plan, roster, registry):
        if not path.is_file():
            raise SupplementPreparationError(f"preflight artifact is missing: {path}")
    handoff = materialize_source_plan(
        plan_path=plan,
        registry_path=registry,
        roster_path=roster,
        output_dir=output_root / "sources/materialized",
        sparse_train=True,
        resume=resume,
        max_workers=max_workers,
        requests_per_second=requests_per_second,
        sparse_http_get=_supplement_alfred_http_get,
    )
    validated = validate_source_handoff(
        handoff_path=handoff,
        plan_path=plan,
        registry_path=registry,
        roster_path=roster,
    )
    summary = {
        "schema_version": "chk4-decision-supplement-source-acquisition-v1",
        "status": "complete",
        "handoff": {"path": str(handoff), "sha256": sha256_file(handoff)},
        "population_count": len(validated["ledgers"]),
        "ready_topic_rows": sum(int(row.get("row_count") or 0) for row in validated["ledgers"]),
        "excluded_topic_rows": sum(
            int(row.get("excluded_sample_count") or 0) for row in validated["ledgers"]
        ),
    }
    _write_json(output_root / "reports/source_acquisition.json", summary)
    _update_root_summary(output_root, "source_acquisition", summary)
    return summary


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SupplementPreparationError("transformers is unavailable") from exc
    if not path.is_dir():
        raise SupplementPreparationError(f"tokenizer path is missing: {path}")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _decimal(value: object, *, label: str) -> Decimal:
    try:
        result = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise SupplementPreparationError(f"invalid numeric value in {label}: {value!r}") from exc
    if not result.is_finite():
        raise SupplementPreparationError(f"non-finite numeric value in {label}")
    return result


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _topic_category(topic: str) -> str | None:
    if topic in PRICE_TOPICS:
        return "prices"
    if topic in ACTIVITY_TOPICS:
        return "employment_activity"
    if topic in FINANCIAL_TOPICS:
        return "financial_conditions"
    return None


def compress_series(series: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_observations = series.get("observations")
    if not isinstance(raw_observations, list):
        return None
    observations: list[tuple[str, Decimal]] = []
    for item in raw_observations:
        if not isinstance(item, Mapping):
            continue
        observation_date = str(item.get("date") or "").strip()
        try:
            date.fromisoformat(observation_date)
            numeric = _decimal(item.get("value"), label=str(series.get("series_id")))
        except (ValueError, SupplementPreparationError):
            continue
        observations.append((observation_date, numeric))
    observations.sort(key=lambda item: item[0])
    if not observations:
        return None
    latest = observations[-1][1]
    changes: list[dict[str, str]] = []
    for lag, name in CHANGE_LAGS:
        if len(observations) > lag:
            changes.append(
                {
                    "relative_horizon": name,
                    "value_change": _decimal_text(latest - observations[-1 - lag][1]),
                }
            )
    title = str(series.get("title") or series.get("series_id") or "").strip()
    series_id = str(series.get("series_id") or "").strip()
    if not title or not series_id:
        return None
    return {
        "series": title,
        "series_code": series_id,
        "frequency": str(series.get("frequency") or "as published").strip(),
        "units": str(series.get("units") or "as published").strip(),
        "latest_value": _decimal_text(latest),
        "relative_changes": changes,
        "_observation_count": len(observations),
        "_latest_observation_date": observations[-1][0],
    }


def compress_indicator_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    topic = str(row.get("indicator") or "").strip()
    category = _topic_category(topic)
    payload = row.get("source_payload")
    if not topic or category is None or not isinstance(payload, Mapping):
        return None
    raw_series = payload.get("series")
    if not isinstance(raw_series, list):
        return None
    compressed = [
        value
        for item in raw_series
        if isinstance(item, Mapping)
        for value in [compress_series(item)]
        if value is not None
    ]
    compressed.sort(
        key=lambda item: (-int(item["_observation_count"]), str(item["series_code"]))
    )
    selected = compressed[:MAX_SERIES_PER_TOPIC]
    if not selected:
        return None
    model_series: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for item in selected:
        model_series.append(
            {key: value for key, value in item.items() if not key.startswith("_")}
        )
        provenance.append(
            {
                "series_code": item["series_code"],
                "observation_count": item["_observation_count"],
                "latest_observation_date": item["_latest_observation_date"],
            }
        )
    return {
        "topic": topic,
        "category": category,
        "series": model_series,
        "provenance": provenance,
    }


def _read_validated_indicator_rows(output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = output_root / "sources/plan/source_plan.json"
    roster = output_root / "sources/contracts/leave_one_out_roster_pre2009.json"
    registry = output_root / "sources/contracts/loo_indicator_sources_pre2009.json"
    handoff = output_root / "sources/materialized/source_handoff.json"
    validated = validate_source_handoff(
        handoff_path=handoff,
        plan_path=plan,
        registry_path=registry,
        roster_path=roster,
    )
    rows: list[dict[str, Any]] = []
    for record in validated["ledgers"]:
        if record.get("split") != "train" or record.get("coverage_mode") != "sparse":
            raise SupplementPreparationError("supplement source handoff is not sparse train-only")
        ledger_dir = Path(str(record["ledger_dir"]))
        rows.extend(_load_jsonl(ledger_dir / "indicator_inputs.jsonl"))
    return rows, validated


def compress_evidence(*, output_root: Path, tokenizer_path: Path) -> dict[str, Any]:
    candidates = {
        str(row["meeting_date"]): row
        for row in _load_jsonl(output_root / "manifests/candidates.jsonl")
    }
    if len(candidates) != EXPECTED_CANDIDATES:
        raise SupplementPreparationError("candidate manifest is incomplete")
    rows, handoff = _read_validated_indicator_rows(output_root)
    by_meeting: dict[str, dict[str, tuple[dict[str, Any], str]]] = defaultdict(dict)
    for row in rows:
        meeting = str(row.get("meeting_date") or "").strip()
        topic = str(row.get("indicator") or "").strip()
        if meeting not in candidates:
            raise SupplementPreparationError(f"ledger meeting is outside supplement: {meeting}")
        expected_vintage = (date.fromisoformat(meeting) - timedelta(days=1)).isoformat()
        if any(
            str(row.get(field) or "") != expected_vintage
            for field in ("requested_vintage_date", "availability_as_of_date", "information_as_of_date")
        ):
            raise SupplementPreparationError(f"D-1 binding failed for {meeting}::{topic}")
        if topic in by_meeting[meeting]:
            raise SupplementPreparationError(f"duplicate meeting/topic ledger row: {meeting}::{topic}")
        compressed = compress_indicator_row(row)
        if compressed is None:
            continue
        for series in compressed["provenance"]:
            if series["latest_observation_date"] > expected_vintage:
                raise SupplementPreparationError(
                    f"future observation leaked for {meeting}::{topic}"
                )
        source_id = str(row.get("source_id") or "").strip()
        if not source_id:
            raise SupplementPreparationError(f"source_id missing for {meeting}::{topic}")
        by_meeting[meeting][topic] = (compressed, source_id)

    tokenizer = _load_tokenizer(tokenizer_path)
    prepared_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    admitted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    action_classes: set[str] = set()
    prompt_token_counts: list[int] = []
    for meeting, candidate in sorted(candidates.items()):
        topic_map = by_meeting.get(meeting, {})
        topic_evidence = []
        source_ids = []
        categories = set()
        provenance = []
        for topic in sorted(topic_map):
            compressed, source_id = topic_map[topic]
            topic_evidence.append(
                {key: value for key, value in compressed.items() if key != "provenance"}
            )
            provenance.append({"topic": topic, "series": compressed["provenance"]})
            source_ids.append(source_id)
            categories.add(str(compressed["category"]))
        reasons = []
        if len(topic_evidence) < MIN_VALID_ATOMIC_TOPICS:
            reasons.append(
                f"valid_topics:{len(topic_evidence)}<{MIN_VALID_ATOMIC_TOPICS}"
            )
        missing_categories = sorted(REQUIRED_CATEGORIES - categories)
        if missing_categories:
            reasons.append("missing_categories:" + ",".join(missing_categories))
        if reasons:
            rejected_rows.append(
                {
                    "schema_version": ADMISSION_SCHEMA,
                    "sample_id": candidate["sample_id"],
                    "meeting_date": meeting,
                    "split": "train",
                    "status": "rejected",
                    "reasons": reasons,
                    "valid_atomic_topic_count": len(topic_evidence),
                    "category_coverage": sorted(categories),
                    "gold": candidate["gold"],
                }
            )
            continue
        model_payload = {"topic_evidence": topic_evidence}
        user_prompt = SUMMARY_USER_PREFIX + canonical_json(model_payload)
        if meeting in user_prompt:
            raise SupplementPreparationError(f"summary prompt leaks meeting identity: {meeting}")
        if any(key in user_prompt.lower() for key in ("meeting_date", "rate_change", "current_rate", "minutes", '"gold"')):
            raise SupplementPreparationError(f"summary prompt contains a forbidden field: {meeting}")
        prompt_tokens = _token_count(tokenizer, SUMMARY_SYSTEM_PROMPT + user_prompt)
        if prompt_tokens > MAX_SUMMARY_INPUT_TOKENS:
            raise SupplementPreparationError(
                f"summary prompt tokens exceed {MAX_SUMMARY_INPUT_TOKENS}: {meeting}={prompt_tokens}"
            )
        input_sha = sha256_text(canonical_json(model_payload))
        prompt_sha = sha256_text(user_prompt)
        sample_id = str(candidate["sample_id"])
        prepared_rows.append(
            {
                "schema_version": PREPARED_SCHEMA,
                "sample_id": sample_id,
                "prompt": user_prompt,
                "input_sha256": input_sha,
                "prompt_sha256": prompt_sha,
            }
        )
        evidence_rows.append(
            {
                "schema_version": EVIDENCE_SCHEMA,
                "sample_id": sample_id,
                "topic_evidence": topic_evidence,
                "input_sha256": input_sha,
            }
        )
        admitted_rows.append(
            {
                "schema_version": ADMISSION_SCHEMA,
                "sample_id": sample_id,
                "meeting_date": meeting,
                "split": "train",
                "population": POPULATION,
                "population_role": "supplement",
                "admission_profile": ADMISSION_PROFILE,
                "status": "admitted",
                "valid_atomic_topic_count": len(topic_evidence),
                "category_coverage": sorted(categories),
                "source_ids": sorted(source_ids),
                "source_provenance": provenance,
                "gold": candidate["gold"],
                "gold_sha256": candidate["gold_sha256"],
                "input_sha256": input_sha,
                "prompt_sha256": prompt_sha,
                "prompt_tokens": prompt_tokens,
            }
        )
        action_classes.add(
            f"{candidate['gold']['direction']}:{candidate['gold']['magnitude_bp']}"
        )
        prompt_token_counts.append(prompt_tokens)

    expected_classes = set(EXPECTED_ACTION_COUNTS)
    release_ok = (
        len(admitted_rows) == MIN_ADMITTED
        and action_classes == expected_classes
    )
    _write_jsonl(output_root / "prepared/summary_requests.jsonl", prepared_rows)
    _write_jsonl(output_root / "prepared/evidence.jsonl", evidence_rows)
    _write_jsonl(output_root / "manifests/admitted.jsonl", admitted_rows)
    _write_jsonl(output_root / "manifests/rejected.jsonl", rejected_rows)
    audit = {
        "schema_version": "chk4-decision-supplement-evidence-audit-v2",
        "status": "complete" if release_ok else "blocked",
        "admission_profile": ADMISSION_PROFILE,
        "candidate_count": len(candidates),
        "admitted_count": len(admitted_rows),
        "rejected_count": len(rejected_rows),
        "minimum_admitted": MIN_ADMITTED,
        "minimum_valid_atomic_topics": MIN_VALID_ATOMIC_TOPICS,
        "required_categories": sorted(REQUIRED_CATEGORIES),
        "action_classes": sorted(action_classes),
        "expected_action_classes": sorted(expected_classes),
        "summary_prompt_tokens": {
            "max": max(prompt_token_counts, default=0),
            "min": min(prompt_token_counts, default=0),
        },
        "source_handoff_payload_sha256": handoff["payload_sha256"],
        "forbidden_current_vintage_inputs": True,
        "meeting_identity_removed_from_model_input": True,
    }
    _write_json(output_root / "reports/evidence_audit.json", audit)
    _update_root_summary(output_root, "evidence", audit)
    if not release_ok:
        raise SupplementPreparationError(
            f"supplement admission failed: admitted={len(admitted_rows)}, classes={sorted(action_classes)}"
        )
    return audit


def _update_root_summary(output_root: Path, stage: str, payload: Mapping[str, Any]) -> None:
    path = output_root / "summary.json"
    current = _load_json(path) if path.is_file() else {
        "schema_version": "chk4-decision-supplement-pipeline-summary-v1",
        "population": POPULATION,
        "stages": {},
    }
    stages = dict(current.get("stages") or {})
    stages[stage] = dict(payload)
    current["stages"] = stages
    current["status"] = "complete" if stage == "qa" and payload.get("status") == "complete" else "in_progress"
    _write_json(path, current)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Freeze labels and source contracts without network or API calls")
    preflight.add_argument("--decision-workbook", type=Path, default=DEFAULT_WORKBOOK)
    preflight.add_argument("--ffr-history", type=Path, default=DEFAULT_FFR_HISTORY)
    preflight.add_argument("--base-roster", type=Path, default=DEFAULT_BASE_ROSTER)
    preflight.add_argument("--base-registry", type=Path, default=DEFAULT_BASE_REGISTRY)

    acquire = subparsers.add_parser("acquire", help="Acquire and seal sparse exact-D-1 ALFRED ledgers")
    acquire.add_argument("--resume", action="store_true")
    acquire.add_argument("--requests-per-second", type=float, default=2.0)
    acquire.add_argument("--max-workers", type=int, default=2)

    compress = subparsers.add_parser("compress", help="Create date-free deterministic summary inputs")
    compress.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_root = args.output_root.expanduser().resolve()
        if args.command == "preflight":
            result = prepare_release(
                output_root=output_root,
                workbook=args.decision_workbook.expanduser().resolve(),
                ffr_history=args.ffr_history.expanduser().resolve(),
                base_roster=args.base_roster.expanduser().resolve(),
                base_registry=args.base_registry.expanduser().resolve(),
            )
        elif args.command == "acquire":
            result = acquire_sources(
                output_root=output_root,
                resume=args.resume,
                requests_per_second=args.requests_per_second,
                max_workers=args.max_workers,
            )
        else:
            result = compress_evidence(
                output_root=output_root,
                tokenizer_path=args.tokenizer_path.expanduser().resolve(),
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (SupplementPreparationError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False), file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_ACTION_COUNTS",
    "POPULATION",
    "SUMMARY_SYSTEM_PROMPT",
    "SupplementPreparationError",
    "compress_evidence",
    "compress_indicator_row",
    "load_candidates",
    "prepare_release",
]
