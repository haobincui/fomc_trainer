"""Create the immutable 48-row chk4 pre-2009 correction SFT release.

This release is a deterministic, zero-repeat child of the sealed pre-2009
balanced release.  It never changes prompts, responses, labels, validation, or
test rows.  Seven probe rows (and every prompt/source-lineage equivalent) are
removed before deterministic temporal-quantile selection.  The selected rows
are arranged into six authoritative optimizer windows, each containing three
hold, three hike, and two cut examples and examples from both populations.

Publication is create-only.  The output is first built and deeply replayed in a
staging directory and is then installed with ``renameat2(RENAME_NOREPLACE)``.
No model or GPU is used.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from open_r1.trainer.sft_prompt_renderer import (
    render_sft_prompt,
    tokenize_sft_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ID = "chk4_decision_pre2009_correction_sft_v1_20260811"
RELEASE_SCHEMA = "chk4-decision-pre2009-correction-sft-release-v1"
HANDOFF_SCHEMA = "chk4-decision-pre2009-correction-sft-handoff-v1"
DATASET_ROLE = "decision_sft_pre2009_correction"
SAMPLER_TYPE = "manifest_fixed_schedule_v3"
SCHEDULE_SCHEMA = "chk4-decision-pre2009-correction-schedule-row-v1"
SELECTION_SCHEMA = "chk4-decision-pre2009-correction-selection-row-v1"
AUDIT_SCHEMA = "chk4-decision-pre2009-correction-audit-v1"
SELECTION_ALGORITHM = (
    "per-role-direction-magnitude-temporal-midpoint-quantiles-without-replacement-v1"
)
SCHEDULE_ORDER_ALGORITHM = "sha256-salted-source-id-with-frozen-window-cells-v1"
SCHEDULE_ORDER_SALT = "chk4-pre2009-correction-schedule-v1-20260811"

PARENT_RELEASE_ID = "chk4_decision_pre2009_train_balanced_v1_20260811"
PARENT_MANIFEST_SHA256 = (
    "5141e00569b94e4af96f030f46176e9cd409199a6f26ac1c63a2ce335fe1e2d6"
)
DEFAULT_PARENT = REPO_ROOT / "dataset/processed/retrain_v2" / PARENT_RELEASE_ID
DEFAULT_OUTPUT = REPO_ROOT / "dataset/processed/retrain_v2" / RELEASE_ID

HELDOUT_SAMPLE_IDS = (
    "dec-0ad00e5fe7cb7333beafdab1",  # sealed 15-case hold
    "dec-bb7d9c61358a339a9a1f4aa5",  # smoke step-1 hold
    "dec-29de02cb43945c20f838fcf7",  # smoke step-2 hold
    "dec-4dfab939b0a910c949931475",  # smoke step-1 hike
    "dec-8b8d55ea065b662a19cefe88",  # smoke step-2 cut
    "dec-0bc683809351ff9da117027b",  # pre-2009 hold probe
    "dec-62f0a67d390dc9a866611dac",  # pre-2009 hike probe
)
HISTORICAL_FIXED_HELDOUT_IDS = HELDOUT_SAMPLE_IDS[:5]
PREREGISTERED_PRE2009_HELDOUTS = {
    "hold": HELDOUT_SAMPLE_IDS[5],
    "hike": HELDOUT_SAMPLE_IDS[6],
}
HELDOUT_SELECTION_SALT = "corrective-holdout-v1"
HELDOUT_SELECTION_ALGORITHM = (
    "min-sha256(sample_id+'|'+salt)-per-supplement-direction-v1"
)

# The quotas also freeze every action magnitude.  Population role ``supplement``
# is the sealed parent's name for the pre-2009 population.
STRATUM_QUOTAS: Mapping[tuple[str, str, int], int] = {
    ("core", "hold", 0): 6,
    ("core", "hike", 25): 8,
    ("core", "cut", 25): 3,
    ("core", "cut", 100): 1,
    ("supplement", "hold", 0): 12,
    ("supplement", "hike", 25): 7,
    ("supplement", "hike", 50): 3,
    ("supplement", "cut", 25): 3,
    ("supplement", "cut", 50): 3,
    ("supplement", "cut", 75): 2,
}
ROWS = 48
OPTIMIZER_STEPS = 6
EFFECTIVE_BATCH_SIZE = 8
PER_WINDOW_COUNTS = {"hold": 3, "hike": 3, "cut": 2}
POPULATION_COUNTS = {"core": 18, "supplement": 30}
DIRECTION_COUNTS = {"hold": 18, "hike": 18, "cut": 12}
ACTION_COUNTS = {
    ("hold", 0): 18,
    ("hike", 25): 15,
    ("hike", 50): 3,
    ("cut", 25): 6,
    ("cut", 50): 3,
    ("cut", 75): 2,
    ("cut", 100): 1,
}

# Cell A uses two core hikes and two pre-2009 cuts; cell B uses one core
# hike/cut and two pre-2009 hikes plus one pre-2009 cut.  A is interleaved at
# optimizer steps 0 and 3 so population/direction exposure is not aligned with
# the endpoints of the cosine schedule.
WINDOW_TEMPLATE_A: tuple[tuple[str, str], ...] = (
    ("core", "hold"),
    ("supplement", "hold"),
    ("core", "hike"),
    ("supplement", "hike"),
    ("supplement", "hold"),
    ("core", "hike"),
    ("supplement", "cut"),
    ("supplement", "cut"),
)
WINDOW_TEMPLATE_B: tuple[tuple[str, str], ...] = (
    ("core", "hold"),
    ("supplement", "hold"),
    ("core", "hike"),
    ("supplement", "hike"),
    ("supplement", "hold"),
    ("core", "cut"),
    ("supplement", "hike"),
    ("supplement", "cut"),
)
WINDOW_TEMPLATES = (
    WINDOW_TEMPLATE_A,
    WINDOW_TEMPLATE_B,
    WINDOW_TEMPLATE_B,
    WINDOW_TEMPLATE_A,
    WINDOW_TEMPLATE_B,
    WINDOW_TEMPLATE_B,
)


class CorrectionReleaseError(RuntimeError):
    """A parent input or derived correction release failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorrectionReleaseError(message)


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


def _sha256_file(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), f"unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorrectionReleaseError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    try:
        raw_rows = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CorrectionReleaseError(f"cannot read {label}: {path}") from exc
    for line_number, raw in enumerate(raw_rows, 1):
        _require(bool(raw), f"{label}:{line_number}: blank row")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CorrectionReleaseError(
                f"{label}:{line_number}: invalid JSON"
            ) from exc
        _require(isinstance(value, dict), f"{label}:{line_number}: not an object")
        rows.append(value)
    return rows


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes(
        path,
        (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_bytes(
        path,
        b"".join((_canonical_json(dict(row)) + "\n").encode() for row in rows),
    )


def _file_record(path: Path, *, root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if path.suffix == ".jsonl":
        result["rows"] = len(path.read_text(encoding="utf-8").splitlines())
    return result


def _file_records(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), f"release contains symlink: {path}")
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative not in {"release_manifest.json", "handoff.json"}:
            record = _file_record(path, root=root)
            records[record["path"]] = record
    return records


def _tokenizer_files(tokenizer_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
    ):
        path = tokenizer_root / name
        if path.is_file() and not path.is_symlink():
            records[name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    _require("tokenizer_config.json" in records, "tokenizer_config.json missing")
    _require(
        "tokenizer.json" in records or "tokenizer.model" in records,
        "tokenizer vocabulary missing",
    )
    return records


def _percentiles(values: Sequence[int]) -> dict[str, int | float]:
    _require(bool(values), "cannot summarize empty token lengths")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    def clean(value: float) -> int | float:
        rounded = round(value, 3)
        return int(rounded) if rounded.is_integer() else rounded

    return {
        "sum": sum(ordered),
        "min": ordered[0],
        "p50": clean(percentile(0.50)),
        "p90": clean(percentile(0.90)),
        "p95": clean(percentile(0.95)),
        "p99": clean(percentile(0.99)),
        "max": ordered[-1],
    }


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _resolve_parent(value: str | Path) -> Path:
    path = Path(value).expanduser()
    _require(not path.is_symlink(), f"parent must not be a symlink: {path}")
    resolved = path.resolve()
    _require(resolved.is_dir(), f"parent release is missing: {resolved}")
    return resolved


def _verify_parent(parent: Path) -> dict[str, Any]:
    manifest_path = parent / "release_manifest.json"
    _require(
        _sha256_file(manifest_path) == PARENT_MANIFEST_SHA256,
        "sealed pre-2009 parent manifest hash drift",
    )
    try:
        from jobs.retrain_v2.materialize_chk4_pre2009_augmented_release import (
            verify_release as verify_parent_release,
        )

        verified = verify_parent_release(
            parent, expected_manifest_sha256=PARENT_MANIFEST_SHA256
        )
    except Exception as exc:
        raise CorrectionReleaseError(
            f"sealed pre-2009 parent replay failed: {exc}"
        ) from exc
    _require(
        verified.get("release_id") == PARENT_RELEASE_ID,
        "sealed parent release identity drift",
    )
    return verified


def _parent_contracts(
    parent: Path, parent_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay the input/system/tokenizer contracts inherited by this child."""

    input_record = parent_manifest.get("input_contract")
    _require(isinstance(input_record, Mapping), "parent input contract binding missing")
    input_path = parent / "contracts/decision_input_contract.json"
    input_contract = _read_json(input_path, label="parent decision input contract")
    _require(
        input_record.get("path") == "contracts/decision_input_contract.json"
        and input_record.get("schema_version") == input_contract.get("schema_version")
        and input_record.get("contract_sha256")
        == input_contract.get("contract_sha256")
        and input_record.get("file_sha256") == _sha256_file(input_path),
        "parent input contract manifest binding drift",
    )
    unsigned_input = dict(input_contract)
    contract_sha = unsigned_input.pop("contract_sha256", None)
    _require(
        contract_sha == _sha256_text(_canonical_json(unsigned_input)),
        "parent input contract payload hash drift",
    )
    system_prompt = input_contract.get("student_system_prompt")
    system_prompt_sha = input_contract.get("student_system_prompt_sha256")
    _require(
        isinstance(system_prompt, str)
        and bool(system_prompt)
        and system_prompt_sha == _sha256_text(system_prompt),
        "parent student system prompt binding drift",
    )

    parent_releases = parent_manifest.get("parent_releases")
    _require(isinstance(parent_releases, Mapping), "parent release lineage missing")
    core_record = parent_releases.get("core_v3")
    _require(isinstance(core_record, Mapping), "parent core-v3 lineage missing")
    raw_core_path = core_record.get("path")
    _require(isinstance(raw_core_path, str) and raw_core_path, "core-v3 path missing")
    core_root = Path(raw_core_path).expanduser()
    if not core_root.is_absolute():
        core_root = REPO_ROOT / core_root
    core_root = _resolve_parent(core_root)
    core_manifest_path = core_root / "release_manifest.json"
    core_manifest_sha = _sha256_file(core_manifest_path)
    _require(
        core_manifest_sha == core_record.get("manifest_sha256"),
        "parent core-v3 manifest hash drift",
    )
    core_manifest = _read_json(core_manifest_path, label="parent core-v3 manifest")
    _require(
        core_manifest.get("release_id") == core_record.get("release_id"),
        "parent core-v3 release identity drift",
    )
    sources = core_manifest.get("sources")
    _require(isinstance(sources, Mapping), "parent core-v3 sources missing")
    tokenizer_source = sources.get("tokenizer")
    _require(isinstance(tokenizer_source, Mapping), "parent tokenizer source missing")
    tokenizer_path_value = tokenizer_source.get("path")
    tokenizer_files = tokenizer_source.get("files")
    _require(
        isinstance(tokenizer_path_value, str)
        and tokenizer_path_value
        and isinstance(tokenizer_files, Mapping)
        and bool(tokenizer_files),
        "parent tokenizer source binding invalid",
    )
    tokenizer_root = Path(tokenizer_path_value).expanduser()
    if not tokenizer_root.is_absolute():
        tokenizer_root = REPO_ROOT / tokenizer_root
    tokenizer_root = tokenizer_root.resolve()
    _require(
        tokenizer_root.is_dir() and not tokenizer_root.is_symlink(),
        "parent tokenizer path missing or unsafe",
    )
    observed_tokenizer_files = _tokenizer_files(tokenizer_root)
    _require(
        dict(tokenizer_files) == observed_tokenizer_files,
        "parent tokenizer file binding drift",
    )
    core_audit = _read_json(
        core_root / "audits/data_quality.json", label="parent core-v3 data audit"
    )
    token_contract = core_audit.get("token_contract")
    _require(isinstance(token_contract, Mapping), "parent token contract missing")
    _require(
        token_contract.get("tokenizer_files") == observed_tokenizer_files
        and token_contract.get("tokenizer_path") == str(tokenizer_root)
        and token_contract.get("sft_max_length") == 3072
        and token_contract.get("single_bos") is True
        and token_contract.get("final_eos") is True
        and token_contract.get("truncation") is False,
        "parent core-v3 token contract drift",
    )
    tokenizer_bundle_sha = _sha256_text(_canonical_json(observed_tokenizer_files))
    return {
        "input_path": input_path,
        "input_contract": input_contract,
        "input_record": dict(input_record),
        "system_prompt": system_prompt,
        "system_prompt_sha256": system_prompt_sha,
        "core_release_id": core_manifest["release_id"],
        "core_manifest_sha256": core_manifest_sha,
        "tokenizer_root": tokenizer_root,
        "tokenizer_files": observed_tokenizer_files,
        "tokenizer_bundle_sha256": tokenizer_bundle_sha,
        "token_contract": dict(token_contract),
    }


def _parent_source_rows(parent: Path) -> list[dict[str, Any]]:
    return _read_jsonl(
        parent / "manifests/source_unique_train.jsonl",
        label="parent unique source train",
    )


def _stratum(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("population_role")),
        str(row.get("direction")),
        int(row.get("magnitude_bp", -1)),
    )


def _heldout_lineage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {str(row.get("sample_id")): row for row in rows}
    _require(len(by_id) == len(rows), "parent unique source IDs are not unique")
    missing = sorted(set(HELDOUT_SAMPLE_IDS) - set(by_id))
    _require(not missing, f"heldout sample IDs missing from parent: {missing}")
    heldout = [by_id[sample_id] for sample_id in HELDOUT_SAMPLE_IDS]
    prompt_hashes = {str(row.get("prompt_sha256")) for row in heldout}
    source_ids = {
        str(source_id)
        for row in heldout
        for source_id in row.get("source_ids", [])
    }
    _require(len(prompt_hashes) == len(heldout), "heldout prompt hashes are not unique")
    replayed_preregistered: dict[str, str] = {}
    for direction, expected_sample_id in PREREGISTERED_PRE2009_HELDOUTS.items():
        population = [
            row
            for row in rows
            if row.get("population_role") == "supplement"
            and row.get("direction") == direction
        ]
        _require(bool(population), f"empty supplement {direction} holdout population")
        selected = min(
            population,
            key=lambda row: (
                _sha256_text(
                    f"{row.get('sample_id')}|{HELDOUT_SELECTION_SALT}"
                ),
                str(row.get("sample_id")),
            ),
        )
        observed_sample_id = str(selected.get("sample_id"))
        _require(
            observed_sample_id == expected_sample_id,
            f"preregistered supplement {direction} holdout replay drift",
        )
        replayed_preregistered[direction] = observed_sample_id
    return {
        "rows": heldout,
        "sample_ids": set(HELDOUT_SAMPLE_IDS),
        "prompt_sha256": prompt_hashes,
        "source_ids": source_ids,
        "historical_fixed_sample_ids": list(HISTORICAL_FIXED_HELDOUT_IDS),
        "preregistered_pre2009_sample_ids": replayed_preregistered,
    }


def _heldout_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    heldout = _heldout_lineage(rows)
    return {
        "holdout_scope": "correction_stage_only",
        "historically_seen_by_parent": True,
        "global_unseen_evaluation_claim_authorized": False,
        "excluded_lineage_keys": ["sample_id", "prompt_sha256", "source_ids"],
        "sample_ids": list(HELDOUT_SAMPLE_IDS),
        "historical_fixed_artifacts": {
            "sample_ids": heldout["historical_fixed_sample_ids"],
            "provenance": "sealed-prior-pilot-and-grpo-smoke-artifacts-v1",
        },
        "preregistered_pre2009_hash_rank": {
            "algorithm": HELDOUT_SELECTION_ALGORITHM,
            "salt": HELDOUT_SELECTION_SALT,
            "population_role": "supplement",
            "directions": ["hold", "hike"],
            "sample_ids_by_direction": heldout[
                "preregistered_pre2009_sample_ids"
            ],
        },
    }


def _midpoint_indices(population: int, quota: int) -> list[int]:
    _require(0 < quota <= population, "invalid temporal-quantile quota")
    indexes = [((2 * index + 1) * population) // (2 * quota) for index in range(quota)]
    _require(
        len(indexes) == len(set(indexes)) and indexes[-1] < population,
        "temporal-quantile selection produced duplicate/out-of-range indexes",
    )
    return indexes


def select_source_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Replay the frozen no-replacement temporal-quantile selection."""

    heldout = _heldout_lineage(rows)
    eligible: list[Mapping[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id"))
        prompt_sha = str(row.get("prompt_sha256"))
        source_ids = {str(value) for value in row.get("source_ids", [])}
        overlaps = (
            sample_id in heldout["sample_ids"]
            or prompt_sha in heldout["prompt_sha256"]
            or bool(source_ids & heldout["source_ids"])
        )
        if not overlaps:
            eligible.append(row)

    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[_stratum(row)].append(row)
    selected: list[dict[str, Any]] = []
    for stratum, quota in STRATUM_QUOTAS.items():
        ordered = sorted(
            grouped.get(stratum, []),
            key=lambda row: (str(row.get("meeting_date")), str(row.get("sample_id"))),
        )
        _require(
            len(ordered) >= quota,
            f"insufficient eligible rows for stratum {stratum}: {len(ordered)} < {quota}",
        )
        indexes = _midpoint_indices(len(ordered), quota)
        for rank, source_index in enumerate(indexes):
            row = dict(ordered[source_index])
            row.update(
                {
                    "schema_version": SELECTION_SCHEMA,
                    "selection_algorithm": SELECTION_ALGORITHM,
                    "stratum_population_rows": len(ordered),
                    "stratum_quota": quota,
                    "stratum_rank": rank,
                    "stratum_source_index": source_index,
                }
            )
            selected.append(row)
    _require(len(selected) == ROWS, "correction selection must contain 48 rows")
    _require(
        len({str(row["sample_id"]) for row in selected}) == ROWS,
        "correction selection contains duplicate source IDs",
    )
    return selected


def _schedule_rows(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    queues: dict[tuple[str, str], deque[Mapping[str, Any]]] = {}
    for population in ("core", "supplement"):
        for direction in ("hold", "hike", "cut"):
            candidates = sorted(
                (
                    row
                    for row in selected
                    if row.get("population_role") == population
                    and row.get("direction") == direction
                ),
                key=lambda row: (
                    _sha256_text(
                        SCHEDULE_ORDER_SALT + "\0" + str(row.get("sample_id"))
                    ),
                    str(row.get("sample_id")),
                ),
            )
            queues[(population, direction)] = deque(candidates)

    result: list[dict[str, Any]] = []
    for optimizer_step, template in enumerate(WINDOW_TEMPLATES):
        for microbatch_slot, key in enumerate(template):
            _require(bool(queues[key]), f"schedule exhausted {key} at step {optimizer_step}")
            source = queues[key].popleft()
            schedule_index = len(result)
            training_row_id = "row-" + _sha256_text(
                _canonical_json(
                    {
                        "release_id": RELEASE_ID,
                        "schedule_index": schedule_index,
                        "source_sample_id": source["sample_id"],
                    }
                )
            )[:24]
            result.append(
                {
                    "schema_version": SCHEDULE_SCHEMA,
                    "schedule_index": schedule_index,
                    "optimizer_step": optimizer_step,
                    "microbatch_slot": microbatch_slot,
                    "training_row_id": training_row_id,
                    "source_sample_id": source["sample_id"],
                    "source_repeat_index": 0,
                    "source_repeat_total": 1,
                    "population_role": source["population_role"],
                    "direction": source["direction"],
                    "magnitude_bp": source["magnitude_bp"],
                    "prompt_sha256": source["prompt_sha256"],
                    "response_sha256": source["response_sha256"],
                    "gold_sha256": source["gold_sha256"],
                }
            )
    leftovers = {key: len(queue) for key, queue in queues.items() if queue}
    _require(not leftovers, f"selected rows were not scheduled: {leftovers}")
    return result


def _sft_rows(
    schedule: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(row["sample_id"]): row for row in selected}
    return [
        {
            "prompt": by_id[str(row["source_sample_id"])]["prompt"],
            "response": by_id[str(row["source_sample_id"])]["response"],
            "schedule_index": row["schedule_index"],
            "source_sample_id": row["source_sample_id"],
            "training_row_id": row["training_row_id"],
        }
        for row in schedule
    ]


def _token_budget(
    selected: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    contracts: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the exact completion-only SFT token exposure on the parent tokenizer."""

    from transformers import AutoTokenizer

    tokenizer_root = Path(str(contracts["tokenizer_root"])).resolve()
    _require(
        _tokenizer_files(tokenizer_root) == contracts["tokenizer_files"],
        "token budget tokenizer bytes drift",
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    _require(
        tokenizer.eos_token is not None and tokenizer.eos_token_id is not None,
        "token budget tokenizer EOS missing",
    )
    by_id = {str(row["sample_id"]): row for row in selected}
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    full_lengths: list[int] = []
    by_direction: dict[str, Counter[str]] = defaultdict(Counter)
    by_population: dict[str, Counter[str]] = defaultdict(Counter)
    for index, scheduled in enumerate(schedule):
        source = by_id[str(scheduled["source_sample_id"])]
        rendered = render_sft_prompt(
            tokenizer,
            [
                {"role": "system", "content": contracts["system_prompt"]},
                {"role": "user", "content": str(source["prompt"])},
            ],
        )
        prompt_ids = tokenize_sft_text(tokenizer, rendered)
        full_ids = tokenize_sft_text(
            tokenizer,
            rendered + str(source["response"]) + tokenizer.eos_token,
        )
        _require(
            full_ids[: len(prompt_ids)] == prompt_ids,
            f"token budget row {index}: prompt prefix drift",
        )
        completion_length = len(full_ids) - len(prompt_ids)
        _require(completion_length > 0, f"token budget row {index}: empty completion")
        _require(
            len(prompt_ids) <= 2560
            and completion_length <= 1024
            and len(full_ids) <= 3072,
            f"token budget row {index}: configured length overflow",
        )
        _require(
            full_ids[-1] == tokenizer.eos_token_id,
            f"token budget row {index}: final EOS drift",
        )
        prompt_lengths.append(len(prompt_ids))
        completion_lengths.append(completion_length)
        full_lengths.append(len(full_ids))
        for group, key in (
            (by_direction, str(source["direction"])),
            (by_population, str(source["population_role"])),
        ):
            group[key]["rows"] += 1
            group[key]["prompt_tokens"] += len(prompt_ids)
            group[key]["supervised_completion_tokens"] += completion_length
            group[key]["full_input_tokens"] += len(full_ids)

    return {
        "rows": len(schedule),
        "prompt_tokens": _percentiles(prompt_lengths),
        "supervised_completion_tokens": _percentiles(completion_lengths),
        "full_input_tokens": _percentiles(full_lengths),
        "by_direction": {
            key: dict(value) for key, value in sorted(by_direction.items())
        },
        "by_population": {
            key: dict(value) for key, value in sorted(by_population.items())
        },
        "overflow_rows": {
            "prompt_gt_2560": 0,
            "completion_gt_1024": 0,
            "full_gt_3072": 0,
        },
        "tokenizer_bundle_sha256": contracts["tokenizer_bundle_sha256"],
        "loader_parameters": contracts["token_contract"]["loader_parameters"],
        "max_length": 3072,
    }


def _audit(
    selected: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    contracts: Mapping[str, Any],
    heldout_contract: Mapping[str, Any],
) -> dict[str, Any]:
    population = Counter(str(row["population_role"]) for row in selected)
    directions = Counter(str(row["direction"]) for row in selected)
    actions = Counter(
        (str(row["direction"]), int(row["magnitude_bp"])) for row in selected
    )
    windows: list[dict[str, Any]] = []
    for step in range(OPTIMIZER_STEPS):
        window = schedule[step * 8 : (step + 1) * 8]
        windows.append(
            {
                "optimizer_step": step,
                "source_sample_ids": [row["source_sample_id"] for row in window],
                "direction_counts": dict(
                    Counter(str(row["direction"]) for row in window)
                ),
                "population_counts": dict(
                    Counter(str(row["population_role"]) for row in window)
                ),
                "core_action_rows": sum(
                    row["population_role"] == "core" and row["direction"] != "hold"
                    for row in window
                ),
                "pre2009_action_rows": sum(
                    row["population_role"] == "supplement"
                    and row["direction"] != "hold"
                    for row in window
                ),
            }
        )
    selected_by_id = {str(row["sample_id"]): row for row in selected}
    scheduled_dates = [
        str(selected_by_id[str(row["source_sample_id"])]["meeting_date"])
        for row in schedule
    ]
    _require(
        scheduled_dates != sorted(scheduled_dates)
        and scheduled_dates != sorted(scheduled_dates, reverse=True),
        "schedule must not be monotonic by meeting date",
    )
    dates_by_population: dict[str, list[str]] = defaultdict(list)
    years_by_population: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selected:
        population_role = str(row["population_role"])
        meeting_date = str(row["meeting_date"])
        dates_by_population[population_role].append(meeting_date)
        years_by_population[population_role][meeting_date[:4]] += 1
    return {
        "schema_version": AUDIT_SCHEMA,
        "quality_status": "passed",
        "selected_unique_rows": len(selected),
        "physical_rows": len(schedule),
        "max_source_repeat": 1,
        "heldout_sample_ids": list(HELDOUT_SAMPLE_IDS),
        "population_counts": dict(population),
        "direction_counts": dict(directions),
        "action_counts": {
            f"{direction}:{magnitude}": count
            for (direction, magnitude), count in sorted(actions.items())
        },
        "population_date_bounds": {
            population_role: {
                "min": min(dates),
                "max": max(dates),
                "year_counts": dict(sorted(years_by_population[population_role].items())),
            }
            for population_role, dates in sorted(dates_by_population.items())
        },
        "heldout_contract": dict(heldout_contract),
        "schedule_order": {
            "algorithm": SCHEDULE_ORDER_ALGORITHM,
            "salt": SCHEDULE_ORDER_SALT,
            "meeting_dates_monotonic": False,
        },
        "windows": windows,
        "token_budget": _token_budget(selected, schedule, contracts),
    }


def _manifest_payload(
    *,
    root: Path,
    parent: Path,
    audit: Mapping[str, Any],
    contracts: Mapping[str, Any],
    heldout_contract: Mapping[str, Any],
    created_at_utc: str,
    handoff_unsigned: Mapping[str, Any],
) -> dict[str, Any]:
    files = _file_records(root)
    token_contract = {
        **dict(contracts["token_contract"]),
        "inheritance": "runtime-verified-through-sealed-parent-core-v3",
        "parent_release_manifest_sha256": PARENT_MANIFEST_SHA256,
        "core_release_id": contracts["core_release_id"],
        "core_manifest_sha256": contracts["core_manifest_sha256"],
        "tokenizer_bundle_sha256": contracts["tokenizer_bundle_sha256"],
    }
    return {
        "schema_version": RELEASE_SCHEMA,
        "release_id": RELEASE_ID,
        "release_type": "selected_correction_sft_train_only",
        "dataset_role": DATASET_ROLE,
        "created_at_utc": created_at_utc,
        "quality_status": "passed",
        "immutable": True,
        "training_ready": True,
        "canonical_dag_bindable": False,
        "test_is_sealed_evaluation_only": True,
        "population_scope": "core-plus-pre2009-correction-subset",
        "grain": (
            "one unique target-decision-blind meeting brief selected once; "
            "physical train exposure is the manifest-bound six-window schedule"
        ),
        "parent_release": {
            "path": str(parent),
            "release_id": PARENT_RELEASE_ID,
            "manifest_sha256": PARENT_MANIFEST_SHA256,
        },
        "selection_contract": {
            "algorithm": SELECTION_ALGORITHM,
            "heldout_sample_ids": list(HELDOUT_SAMPLE_IDS),
            "heldout_exclusion": ["sample_id", "prompt_sha256", "source_ids"],
            "heldout_contract": "heldout_contract",
            "stratum_quotas": {
                f"{population}:{direction}:{magnitude}": count
                for (population, direction, magnitude), count in STRATUM_QUOTAS.items()
            },
            "selected_unique_rows": ROWS,
            "zero_repeats": True,
        },
        "heldout_contract": dict(heldout_contract),
        "schedule_order_contract": {
            "algorithm": SCHEDULE_ORDER_ALGORITHM,
            "salt": SCHEDULE_ORDER_SALT,
            "window_cells": ["A", "B", "B", "A", "B", "B"],
            "meeting_date_order_forbidden": True,
        },
        "unique_split_counts": {"train": ROWS, "validation": 13, "test": 13},
        "physical_split_counts": {
            "decision_sft": {"train": ROWS, "validation": 13, "test": 13}
        },
        "files": files,
        "sampler_contract": {
            "type": SAMPLER_TYPE,
            "seed": None,
            "schedule_path": "manifests/sampler_schedule.jsonl",
            "schedule_sha256": files["manifests/sampler_schedule.jsonl"]["sha256"],
            "schedule_rows": ROWS,
            "train_path": "decision_sft/train.jsonl",
            "train_sha256": files["decision_sft/train.jsonl"]["sha256"],
            "train_rows": ROWS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "world_size": 1,
            "optimizer_steps": OPTIMIZER_STEPS,
            "shuffle_dataset": False,
            "direction_counts": DIRECTION_COUNTS,
            "per_optimizer_window": PER_WINDOW_COUNTS,
            "population_counts": POPULATION_COUNTS,
            "per_optimizer_window_population": {"core": 3, "supplement": 5},
            "order_is_authoritative": True,
            "all_unique_sources_covered": True,
            "optimizer_windows_with_duplicate_source": 0,
            "source_population": "pre2009_correction_unique_train_only",
            "oversampling_layers": 0,
            "max_source_repeat": 1,
            "repeat_histogram": {"1": ROWS},
            "required_sampler": "fixed_sequential_schedule_index_v3",
            "secondary_shuffle_forbidden": True,
        },
        "training_role": {
            "dataset_path": "decision_sft",
            "completion_only_loss": True,
            "max_length": 3072,
            "parent_role": "selected_pre2009_cp38_exact_merged",
            "sampler_contract": "sampler_contract",
        },
        "input_contract": {
            "path": "contracts/decision_input_contract.json",
            "schema_version": contracts["input_contract"]["schema_version"],
            "contract_sha256": contracts["input_contract"]["contract_sha256"],
            "file_sha256": files["contracts/decision_input_contract.json"]["sha256"],
            "inherited_byte_for_byte": True,
            "parent_release_manifest_sha256": PARENT_MANIFEST_SHA256,
        },
        "system_prompt_contract": {
            "source": "contracts/decision_input_contract.json",
            "student_system_prompt_sha256": contracts["system_prompt_sha256"],
            "inherited_unchanged": True,
        },
        "token_contract": token_contract,
        "evaluation_inheritance": {
            split: {
                "source_path": str(parent / f"decision_sft/{split}.jsonl"),
                "copied_byte_for_byte": True,
                "sha256": files[f"decision_sft/{split}.jsonl"]["sha256"],
                "rows": files[f"decision_sft/{split}.jsonl"]["rows"],
            }
            for split in ("validation", "test")
        },
        "audit": {
            "path": "audits/data_quality.json",
            "sha256": files["audits/data_quality.json"]["sha256"],
            "quality_status": audit["quality_status"],
        },
        "implementation": {
            "materializer_snapshot": "provenance/materializer_snapshot.py",
            "materializer_snapshot_sha256": files[
                "provenance/materializer_snapshot.py"
            ]["sha256"],
        },
        "handoff": {
            "path": "handoff.json",
            "schema_version": HANDOFF_SCHEMA,
            "unsigned_payload_sha256": _sha256_text(
                _canonical_json(dict(handoff_unsigned))
            ),
        },
        "limitations": {
            "core_cut_eligible_population_fully_exposed": True,
            "unseen_core_cut_gate_available": False,
            "action_generalization_claim_authorized": False,
        },
    }


def build_release(parent: Path, staging: Path) -> dict[str, Any]:
    parent = _resolve_parent(parent)
    verified_parent = _verify_parent(parent)
    contracts = _parent_contracts(parent, verified_parent)
    rows = _parent_source_rows(parent)
    heldout_contract = _heldout_contract(rows)
    selected = select_source_rows(rows)
    schedule = _schedule_rows(selected)
    sft_rows = _sft_rows(schedule, selected)
    audit = _audit(selected, schedule, contracts, heldout_contract)

    _require(not staging.exists() and not staging.is_symlink(), "staging path occupied")
    staging.mkdir(parents=True)
    _write_jsonl(staging / "manifests/source_selection.jsonl", selected)
    heldout_by_id = {str(row["sample_id"]): row for row in rows}
    _write_jsonl(
        staging / "manifests/heldout_probe.jsonl",
        [
            {
                "sample_id": sample_id,
                "population_role": heldout_by_id[sample_id]["population_role"],
                "direction": heldout_by_id[sample_id]["direction"],
                "magnitude_bp": heldout_by_id[sample_id]["magnitude_bp"],
                "meeting_date": heldout_by_id[sample_id]["meeting_date"],
                "prompt_sha256": heldout_by_id[sample_id]["prompt_sha256"],
                "source_ids": heldout_by_id[sample_id]["source_ids"],
                "holdout_provenance": (
                    "sealed-prior-pilot-and-grpo-smoke-artifacts-v1"
                    if sample_id in HISTORICAL_FIXED_HELDOUT_IDS
                    else HELDOUT_SELECTION_ALGORITHM
                ),
            }
            for sample_id in HELDOUT_SAMPLE_IDS
        ],
    )
    _write_jsonl(staging / "manifests/sampler_schedule.jsonl", schedule)
    _write_jsonl(staging / "decision_sft/train.jsonl", sft_rows)
    for split in ("validation", "test"):
        destination = staging / f"decision_sft/{split}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / f"decision_sft/{split}.jsonl", destination)
    input_destination = staging / "contracts/decision_input_contract.json"
    input_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(contracts["input_path"], input_destination)
    _write_json(staging / "audits/data_quality.json", audit)
    snapshot = staging / "provenance/materializer_snapshot.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).resolve(), snapshot)
    created_at_utc = _utc_now()
    handoff_unsigned = {
        "schema_version": HANDOFF_SCHEMA,
        "release_id": RELEASE_ID,
        "created_at_utc": created_at_utc,
        "dataset_role": DATASET_ROLE,
        "quality_status": "passed",
        "immutable": True,
        "training_ready": True,
        "release_manifest": "release_manifest.json",
        "dataset_path": "decision_sft",
        "sampler_schedule": "manifests/sampler_schedule.jsonl",
        "physical_train_rows": ROWS,
        "optimizer_steps": OPTIMIZER_STEPS,
        "heldout_contract_sha256": _sha256_text(
            _canonical_json(heldout_contract)
        ),
        "test_is_sealed_evaluation_only": True,
    }
    manifest = _manifest_payload(
        root=staging,
        parent=parent,
        audit=audit,
        contracts=contracts,
        heldout_contract=heldout_contract,
        created_at_utc=created_at_utc,
        handoff_unsigned=handoff_unsigned,
    )
    _write_json(staging / "release_manifest.json", manifest)
    manifest_sha = _sha256_file(staging / "release_manifest.json")
    _write_json(
        staging / "handoff.json",
        {**handoff_unsigned, "release_manifest_sha256": manifest_sha},
    )
    return manifest


def _record_path(root: Path, record: Mapping[str, Any], *, label: str) -> Path:
    relative = record.get("path")
    _require(
        isinstance(relative, str) and relative and not Path(relative).is_absolute(),
        f"{label} path invalid",
    )
    candidate = root / relative
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        f"missing or unsafe {label}: {candidate}",
    )
    path = candidate.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise CorrectionReleaseError(f"{label} path escapes release") from exc
    _require(path.is_file(), f"missing {label}: {path}")
    return path


def verify_release(
    root: Path,
    *,
    expected_manifest_sha256: str,
    require_release_name: bool = True,
) -> dict[str, Any]:
    """Deeply replay an existing correction release and return its manifest."""

    raw_release = root.expanduser()
    _require(not raw_release.is_symlink(), "correction release root must not be a symlink")
    release = raw_release.resolve()
    _require(release.is_dir(), "invalid release root")
    if require_release_name:
        _require(release.name == RELEASE_ID, "correction release directory drift")
    manifest_path = release / "release_manifest.json"
    observed_manifest_sha = _sha256_file(manifest_path)
    _require(
        isinstance(expected_manifest_sha256, str)
        and len(expected_manifest_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_manifest_sha256),
        "correction release requires an external manifest SHA-256 pin",
    )
    _require(
        observed_manifest_sha == expected_manifest_sha256,
        "correction release manifest hash drift",
    )
    manifest = _read_json(manifest_path, label="correction release manifest")
    _require(
        manifest.get("schema_version") == RELEASE_SCHEMA
        and manifest.get("release_id") == RELEASE_ID
        and manifest.get("dataset_role") == DATASET_ROLE
        and manifest.get("quality_status") == "passed"
        and manifest.get("immutable") is True
        and manifest.get("training_ready") is True
        and manifest.get("canonical_dag_bindable") is False
        and manifest.get("test_is_sealed_evaluation_only") is True,
        "correction release identity/readiness drift",
    )
    parent_record = manifest.get("parent_release")
    _require(isinstance(parent_record, Mapping), "parent release binding missing")
    _require(
        parent_record.get("release_id") == PARENT_RELEASE_ID
        and parent_record.get("manifest_sha256") == PARENT_MANIFEST_SHA256,
        "parent release binding drift",
    )
    parent = _resolve_parent(str(parent_record.get("path")))
    verified_parent = _verify_parent(parent)
    contracts = _parent_contracts(parent, verified_parent)

    files = manifest.get("files")
    _require(isinstance(files, Mapping), "release files binding missing")
    release_members = list(release.rglob("*"))
    _require(
        all(not path.is_symlink() for path in release_members),
        "release contains a symlink",
    )
    expected_files = {
        path.relative_to(release).as_posix()
        for path in release_members
        if path.is_file()
        and path.relative_to(release).as_posix()
        not in {"release_manifest.json", "handoff.json"}
    }
    _require(set(files) == expected_files, "release file inventory drift")
    for name, raw_record in files.items():
        _require(isinstance(raw_record, Mapping), f"invalid file record: {name}")
        path = _record_path(release, raw_record, label=name)
        _require(path.relative_to(release).as_posix() == name, f"file path drift: {name}")
        _require(_sha256_file(path) == raw_record.get("sha256"), f"file hash drift: {name}")
        _require(path.stat().st_size == raw_record.get("bytes"), f"file size drift: {name}")
        if path.suffix == ".jsonl":
            _require(
                len(_read_jsonl(path, label=name)) == raw_record.get("rows"),
                f"file row count drift: {name}",
            )

    parent_rows = _parent_source_rows(parent)
    expected_heldout_contract = _heldout_contract(parent_rows)
    expected_selection = select_source_rows(parent_rows)
    selected = _read_jsonl(
        release / "manifests/source_selection.jsonl", label="source selection"
    )
    _require(selected == expected_selection, "source selection replay drift")
    parent_by_id = {str(row["sample_id"]): row for row in parent_rows}
    expected_heldout_rows = [
        {
            "sample_id": sample_id,
            "population_role": parent_by_id[sample_id]["population_role"],
            "direction": parent_by_id[sample_id]["direction"],
            "magnitude_bp": parent_by_id[sample_id]["magnitude_bp"],
            "meeting_date": parent_by_id[sample_id]["meeting_date"],
            "prompt_sha256": parent_by_id[sample_id]["prompt_sha256"],
            "source_ids": parent_by_id[sample_id]["source_ids"],
            "holdout_provenance": (
                "sealed-prior-pilot-and-grpo-smoke-artifacts-v1"
                if sample_id in HISTORICAL_FIXED_HELDOUT_IDS
                else HELDOUT_SELECTION_ALGORITHM
            ),
        }
        for sample_id in HELDOUT_SAMPLE_IDS
    ]
    heldout_rows = _read_jsonl(
        release / "manifests/heldout_probe.jsonl", label="heldout probe manifest"
    )
    _require(
        heldout_rows == expected_heldout_rows,
        "heldout probe semantic replay drift",
    )
    expected_schedule = _schedule_rows(selected)
    schedule = _read_jsonl(
        release / "manifests/sampler_schedule.jsonl", label="sampler schedule"
    )
    _require(schedule == expected_schedule, "correction schedule replay drift")
    expected_train = _sft_rows(schedule, selected)
    train = _read_jsonl(release / "decision_sft/train.jsonl", label="SFT train")
    _require(train == expected_train, "correction SFT train reconstruction drift")

    heldout = _heldout_lineage(parent_rows)
    selected_ids = {str(row["sample_id"]) for row in selected}
    selected_prompts = {str(row["prompt_sha256"]) for row in selected}
    selected_sources = {
        str(source_id)
        for row in selected
        for source_id in row.get("source_ids", [])
    }
    _require(not selected_ids & heldout["sample_ids"], "heldout sample overlap")
    _require(not selected_prompts & heldout["prompt_sha256"], "heldout prompt overlap")
    _require(not selected_sources & heldout["source_ids"], "heldout source overlap")

    stratum_counts = Counter(_stratum(row) for row in selected)
    _require(dict(stratum_counts) == dict(STRATUM_QUOTAS), "stratum quota drift")
    _require(
        Counter(str(row["population_role"]) for row in selected)
        == Counter(POPULATION_COUNTS),
        "population counts drift",
    )
    _require(
        Counter(str(row["direction"]) for row in selected)
        == Counter(DIRECTION_COUNTS),
        "direction counts drift",
    )
    _require(
        Counter(
            (str(row["direction"]), int(row["magnitude_bp"])) for row in selected
        )
        == Counter(ACTION_COUNTS),
        "action magnitude counts drift",
    )
    for step in range(OPTIMIZER_STEPS):
        window = schedule[step * 8 : (step + 1) * 8]
        _require(
            Counter(str(row["direction"]) for row in window)
            == Counter(PER_WINDOW_COUNTS),
            f"optimizer window {step} direction drift",
        )
        _require(
            Counter(str(row["population_role"]) for row in window)
            == Counter({"core": 3, "supplement": 5}),
            f"optimizer window {step} population drift",
        )
        _require(
            len({str(row["source_sample_id"]) for row in window}) == 8,
            f"optimizer window {step} duplicate source",
        )
        for population in ("core", "supplement"):
            _require(
                any(
                    row["population_role"] == population
                    and row["direction"] != "hold"
                    for row in window
                ),
                f"optimizer window {step} lacks {population} action",
            )

    for split in ("validation", "test"):
        child = release / f"decision_sft/{split}.jsonl"
        source = parent / f"decision_sft/{split}.jsonl"
        _require(_sha256_file(child) == _sha256_file(source), f"{split} inheritance drift")

    _require(
        manifest.get("grain")
        == (
            "one unique target-decision-blind meeting brief selected once; "
            "physical train exposure is the manifest-bound six-window schedule"
        ),
        "release grain drift",
    )
    _require(
        manifest.get("unique_split_counts")
        == {"train": ROWS, "validation": 13, "test": 13},
        "unique split counts drift",
    )
    _require(
        manifest.get("physical_split_counts")
        == {
            "decision_sft": {"train": ROWS, "validation": 13, "test": 13}
        },
        "physical split counts drift",
    )
    _require(
        manifest.get("heldout_contract") == expected_heldout_contract,
        "heldout contract replay drift",
    )
    _require(
        manifest.get("selection_contract")
        == {
            "algorithm": SELECTION_ALGORITHM,
            "heldout_sample_ids": list(HELDOUT_SAMPLE_IDS),
            "heldout_exclusion": ["sample_id", "prompt_sha256", "source_ids"],
            "heldout_contract": "heldout_contract",
            "stratum_quotas": {
                f"{population}:{direction}:{magnitude}": count
                for (population, direction, magnitude), count in STRATUM_QUOTAS.items()
            },
            "selected_unique_rows": ROWS,
            "zero_repeats": True,
        },
        "selection contract drift",
    )
    child_input = release / "contracts/decision_input_contract.json"
    _require(
        child_input.read_bytes() == Path(contracts["input_path"]).read_bytes(),
        "decision input contract is not byte-inherited",
    )
    _require(
        manifest.get("input_contract")
        == {
            "path": "contracts/decision_input_contract.json",
            "schema_version": contracts["input_contract"]["schema_version"],
            "contract_sha256": contracts["input_contract"]["contract_sha256"],
            "file_sha256": _sha256_file(child_input),
            "inherited_byte_for_byte": True,
            "parent_release_manifest_sha256": PARENT_MANIFEST_SHA256,
        },
        "input contract inheritance binding drift",
    )
    _require(
        manifest.get("system_prompt_contract")
        == {
            "source": "contracts/decision_input_contract.json",
            "student_system_prompt_sha256": contracts["system_prompt_sha256"],
            "inherited_unchanged": True,
        },
        "system prompt inheritance binding drift",
    )
    expected_token_contract = {
        **dict(contracts["token_contract"]),
        "inheritance": "runtime-verified-through-sealed-parent-core-v3",
        "parent_release_manifest_sha256": PARENT_MANIFEST_SHA256,
        "core_release_id": contracts["core_release_id"],
        "core_manifest_sha256": contracts["core_manifest_sha256"],
        "tokenizer_bundle_sha256": contracts["tokenizer_bundle_sha256"],
    }
    _require(
        manifest.get("token_contract") == expected_token_contract,
        "token contract inheritance binding drift",
    )
    expected_evaluation = {
        split: {
            "source_path": str(parent / f"decision_sft/{split}.jsonl"),
            "copied_byte_for_byte": True,
            "sha256": _sha256_file(parent / f"decision_sft/{split}.jsonl"),
            "rows": len(
                _read_jsonl(
                    parent / f"decision_sft/{split}.jsonl",
                    label=f"parent {split} SFT",
                )
            ),
        }
        for split in ("validation", "test")
    }
    _require(
        manifest.get("evaluation_inheritance") == expected_evaluation,
        "evaluation inheritance record drift",
    )

    expected_sampler = {
        "type": SAMPLER_TYPE,
        "seed": None,
        "schedule_path": "manifests/sampler_schedule.jsonl",
        "schedule_sha256": _sha256_file(release / "manifests/sampler_schedule.jsonl"),
        "schedule_rows": ROWS,
        "train_path": "decision_sft/train.jsonl",
        "train_sha256": _sha256_file(release / "decision_sft/train.jsonl"),
        "train_rows": ROWS,
        "effective_batch_size": 8,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "world_size": 1,
        "optimizer_steps": 6,
        "shuffle_dataset": False,
        "direction_counts": DIRECTION_COUNTS,
        "per_optimizer_window": PER_WINDOW_COUNTS,
        "population_counts": POPULATION_COUNTS,
        "per_optimizer_window_population": {"core": 3, "supplement": 5},
        "order_is_authoritative": True,
        "all_unique_sources_covered": True,
        "optimizer_windows_with_duplicate_source": 0,
        "source_population": "pre2009_correction_unique_train_only",
        "oversampling_layers": 0,
        "max_source_repeat": 1,
        "repeat_histogram": {"1": ROWS},
        "required_sampler": "fixed_sequential_schedule_index_v3",
        "secondary_shuffle_forbidden": True,
    }
    _require(manifest.get("sampler_contract") == expected_sampler, "sampler contract drift")
    _require(
        manifest.get("training_role")
        == {
            "dataset_path": "decision_sft",
            "completion_only_loss": True,
            "max_length": 3072,
            "parent_role": "selected_pre2009_cp38_exact_merged",
            "sampler_contract": "sampler_contract",
        },
        "training role contract drift",
    )
    _require(
        manifest.get("schedule_order_contract")
        == {
            "algorithm": SCHEDULE_ORDER_ALGORITHM,
            "salt": SCHEDULE_ORDER_SALT,
            "window_cells": ["A", "B", "B", "A", "B", "B"],
            "meeting_date_order_forbidden": True,
        },
        "schedule order contract drift",
    )
    materializer_snapshot = release / "provenance/materializer_snapshot.py"
    _require(
        materializer_snapshot.read_bytes() == Path(__file__).resolve().read_bytes(),
        "materializer snapshot/source drift",
    )
    _require(
        manifest.get("implementation")
        == {
            "materializer_snapshot": "provenance/materializer_snapshot.py",
            "materializer_snapshot_sha256": _sha256_file(materializer_snapshot),
        },
        "materializer implementation binding drift",
    )
    _require(
        manifest.get("limitations")
        == {
            "core_cut_eligible_population_fully_exposed": True,
            "unseen_core_cut_gate_available": False,
            "action_generalization_claim_authorized": False,
        },
        "release limitations disclosure drift",
    )
    audit = _read_json(release / "audits/data_quality.json", label="data audit")
    _require(
        manifest.get("audit")
        == {
            "path": "audits/data_quality.json",
            "sha256": _sha256_file(release / "audits/data_quality.json"),
            "quality_status": "passed",
        },
        "data audit manifest binding drift",
    )
    _require(
        audit
        == _audit(
            selected,
            schedule,
            contracts,
            expected_heldout_contract,
        ),
        "data audit replay drift",
    )
    handoff = _read_json(release / "handoff.json", label="correction handoff")
    unsigned_handoff = dict(handoff)
    handoff_manifest_sha = unsigned_handoff.pop("release_manifest_sha256", None)
    _require(
        manifest.get("handoff")
        == {
            "path": "handoff.json",
            "schema_version": HANDOFF_SCHEMA,
            "unsigned_payload_sha256": _sha256_text(
                _canonical_json(unsigned_handoff)
            ),
        },
        "handoff unsigned payload binding drift",
    )
    _require(
        unsigned_handoff
        == {
            "schema_version": HANDOFF_SCHEMA,
            "release_id": RELEASE_ID,
            "created_at_utc": manifest.get("created_at_utc"),
            "dataset_role": DATASET_ROLE,
            "quality_status": "passed",
            "immutable": True,
            "training_ready": True,
            "release_manifest": "release_manifest.json",
            "dataset_path": "decision_sft",
            "sampler_schedule": "manifests/sampler_schedule.jsonl",
            "physical_train_rows": ROWS,
            "optimizer_steps": OPTIMIZER_STEPS,
            "heldout_contract_sha256": _sha256_text(
                _canonical_json(expected_heldout_contract)
            ),
            "test_is_sealed_evaluation_only": True,
        }
        and handoff_manifest_sha == observed_manifest_sha,
        "handoff contract drift",
    )
    return {**manifest, "release_manifest_sha256": observed_manifest_sha}


def verify_runtime_release(
    *,
    dataset_dir: str | Path,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    dataset_role: str,
    system_prompt: str | None,
    model_path: str | Path,
) -> dict[str, Any]:
    """Bind the correction release to the custom correction SFT runtime."""

    from open_r1.trainer.dataset_release import (
        CHK4_PRE2009_SFT_ROLE,
        CHK4_STUDENT_SYSTEM_PROMPT,
        CHK4_STUDENT_SYSTEM_PROMPT_SHA256,
        verify_chk4_pre2009_augmented_release,
    )

    _require(dataset_role == DATASET_ROLE, "correction dataset role drift")
    _require(
        system_prompt == CHK4_STUDENT_SYSTEM_PROMPT,
        "correction system prompt drift",
    )
    raw_manifest = Path(manifest_path).expanduser()
    raw_dataset = Path(dataset_dir).expanduser()
    _require(
        not raw_manifest.is_symlink() and not raw_dataset.is_symlink(),
        "correction runtime paths must not be symlinks",
    )
    manifest = raw_manifest.resolve()
    dataset = raw_dataset.resolve()
    root = manifest.parent
    _require(dataset == root / "decision_sft", "correction dataset path drift")
    verified = verify_release(
        root,
        expected_manifest_sha256=expected_manifest_sha256,
        require_release_name=True,
    )
    parent = _resolve_parent(str(verified["parent_release"]["path"]))
    parent_runtime = verify_chk4_pre2009_augmented_release(
        dataset_dir=parent / "decision_sft",
        manifest_path=parent / "release_manifest.json",
        expected_manifest_sha256=PARENT_MANIFEST_SHA256,
        dataset_role=CHK4_PRE2009_SFT_ROLE,
        system_prompt=system_prompt,
        model_path=model_path,
    )
    sampler = dict(verified["sampler_contract"])
    sampler["schedule_path"] = str(root / str(sampler["schedule_path"]))
    sampler["train_path"] = str(root / str(sampler["train_path"]))
    return {
        "schema_version": "chk4-pre2009-correction-runtime-binding-v1",
        "release_id": RELEASE_ID,
        "dataset_role": DATASET_ROLE,
        "physical_dataset_role": "decision_sft",
        "release_manifest_path": str(manifest),
        "release_manifest_sha256": expected_manifest_sha256,
        "split_files": {
            "train": root / "decision_sft/train.jsonl",
            "validation": root / "decision_sft/validation.jsonl",
        },
        "test_verified_but_not_loaded": True,
        "system_prompt_sha256": CHK4_STUDENT_SYSTEM_PROMPT_SHA256,
        "tokenizer_binding": dict(parent_runtime["tokenizer_binding"]),
        "sampler_contract": sampler,
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CorrectionReleaseError("renameat2 is unavailable; cannot publish safely")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(error, os.strerror(error), str(destination))


def _seal_tree(root: Path, *, seal_root: bool = True) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        _require(not path.is_symlink(), f"cannot seal symlink: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555 if seal_root else 0o700)


def _cleanup_staging(staging_root: Path) -> None:
    """Remove only this publisher's validated temporary staging directory."""

    expected_prefix = f".{RELEASE_ID}.staging-"
    _require(
        staging_root.parent.is_dir()
        and staging_root.name.startswith(expected_prefix)
        and staging_root != staging_root.parent,
        f"refusing unsafe staging cleanup target: {staging_root}",
    )
    if not staging_root.exists():
        return
    for path in sorted(staging_root.rglob("*"), reverse=True):
        _require(not path.is_symlink(), f"staging cleanup found symlink: {path}")
        path.chmod(0o700 if path.is_dir() else 0o600)
    staging_root.chmod(0o700)
    shutil.rmtree(staging_root)


def materialize(parent: Path, output: Path) -> dict[str, Any]:
    destination = output.expanduser().resolve()
    _require(not destination.exists() and not destination.is_symlink(), "output exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{RELEASE_ID}.staging-", dir=destination.parent)
    )
    staging = staging_root / RELEASE_ID
    try:
        build_release(parent, staging)
        manifest_sha = _sha256_file(staging / "release_manifest.json")
        verify_release(
            staging,
            expected_manifest_sha256=manifest_sha,
            require_release_name=True,
        )
        # Seal every member before publication, but retain write permission on
        # the release root itself because this filesystem rejects renameat2 for
        # a mode-0555 source directory.  The enclosing staging directory is
        # private (0700), so no other process can mutate the sealed members.
        _seal_tree(staging, seal_root=False)
        _rename_noreplace(staging, destination)
        destination.chmod(0o555)
        staging_root.rmdir()
    except Exception:
        _cleanup_staging(staging_root)
        raise
    return verify_release(destination, expected_manifest_sha256=manifest_sha)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        parent = _resolve_parent(args.parent)
        _verify_parent(parent)
        selected = select_source_rows(_parent_source_rows(parent))
        schedule = _schedule_rows(selected)
        if args.execute:
            result = materialize(parent, args.output)
            status = "published"
        else:
            result = {
                "release_id": RELEASE_ID,
                "dataset_role": DATASET_ROLE,
                "selected_rows": len(selected),
                "schedule_rows": len(schedule),
                "output": str(args.output.expanduser().resolve()),
                "create_only": True,
            }
            status = "ready"
        print(json.dumps({"status": status, **result}, ensure_ascii=False, indent=2))
        return 0
    except (CorrectionReleaseError, FileExistsError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
