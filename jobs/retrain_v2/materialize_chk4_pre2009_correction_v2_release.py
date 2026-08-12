"""Publish and replay the immutable chk4 correction-v2 SFT release.

The child is additive and never mutates the sealed pre-2009 parent or the
correction-v1 release.  Fourteen prompts whose labels have already influenced
earlier chk4 model selection are moved into training (teacher targets always
come byte-for-byte from the sealed parent).  Twenty-one separately committed
prompts remain excluded for selection, blind confirmation, smoke, and
retention.  The final schedule has 48 physical rows backed by 33 unique source
rows and is consumed only by the independent correction-v2 sampler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.retrain_v2 import (
    materialize_chk4_pre2009_correction_release as legacy,
)
from open_r1.provenance import fingerprint_artifact_path


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ID = "chk4_decision_pre2009_correction_sft_v2_20260811"
RELEASE_SCHEMA = "chk4-decision-pre2009-correction-sft-release-v2"
HANDOFF_SCHEMA = "chk4-decision-pre2009-correction-sft-handoff-v2"
DATASET_ROLE = "decision_sft_pre2009_correction_v2"
SAMPLER_TYPE = "manifest_fixed_schedule_correction_v2"
SCHEDULE_SCHEMA = "chk4-decision-pre2009-correction-v2-schedule-row-v1"
SELECTION_SCHEMA = "chk4-decision-pre2009-correction-v2-selection-row-v1"
PARTITION_SCHEMA = "chk4-correction-v2-partition-commitment-v1"
AUDIT_SCHEMA = "chk4-decision-pre2009-correction-v2-audit-v1"

PARENT_RELEASE_ID = legacy.PARENT_RELEASE_ID
PARENT_MANIFEST_SHA256 = legacy.PARENT_MANIFEST_SHA256
DEFAULT_PARENT = legacy.DEFAULT_PARENT
DEFAULT_OUTPUT = REPO_ROOT / "dataset/processed/retrain_v2" / RELEASE_ID
EXPECTED_PYTHON = Path("/home/haobin_cui/.conda/envs/fomc_trainer/bin/python")
EXPECTED_PYTHON_VERSION = (3, 10, 9)
EXPECTED_PACKAGE_VERSIONS = {
    "accelerate": "1.4.0",
    "datasets": "4.8.4",
    "peft": "0.15.2",
    "tokenizers": "0.22.2",
    "torch": "2.10.0+cu128",
    "transformers": "4.57.6",
    "trl": "1.2.0",
}

V1_RELEASE = (
    REPO_ROOT
    / "dataset/processed/retrain_v2"
    / "chk4_decision_pre2009_correction_sft_v1_20260811"
)
V1_RELEASE_MANIFEST_SHA256 = (
    "1930991a24ce4cd615b0e96ef233991e2885b3ee9a45423686e1005181be0713"
)
V1_TRAIN_SHA256 = "b8285febc620637eabef29876747b92bbea3d7fca933f0752781b2686eabc06f"

TRAINING_PARENT_MODEL = (
    REPO_ROOT / "output/training/retrain_v2/"
    "chk4_from_chk1_cp200_sft_grpo_pre2009_balanced_lr1e5_steps39_v1_20260811/"
    "selected_sft_checkpoints/checkpoint-38/merged/chk4_sft"
)
TRAINING_PARENT_SHA256 = (
    "715bdb763043e7fda49e4a189b601d5e070fefbcecd14388bde41dbf7c165dd0"
)

BURNED_SAMPLE_IDS = (
    "dec-0ad00e5fe7cb7333beafdab1",
    "dec-0bc683809351ff9da117027b",
    "dec-1f250f67a099c1ef6a953ea3",
    "dec-29de02cb43945c20f838fcf7",
    "dec-348b7c1193c8764b2adca770",
    "dec-4596d7944cbdef59dd5a4f99",
    "dec-4dfab939b0a910c949931475",
    "dec-5848e27930ae03a1fa6bd5cb",
    "dec-62f0a67d390dc9a866611dac",
    "dec-8b8d55ea065b662a19cefe88",
    "dec-bb7d9c61358a339a9a1f4aa5",
    "dec-de9d3a26759a82709f068f65",
    "dec-ed9ce1ed2c210798357e0bb5",
    "dec-f66acacb4f43fea1e06bbfe8",
)
BURNED_SET_SHA256 = "81d75b2f4ec2c0104bab4960bf80788969d43422e650a14d426cf0d602f9cf5a"

WITNESS_ARTIFACTS: Mapping[str, tuple[str, str]] = {
    "core_steps24_pilot": (
        "output/eval/retrain_v2/"
        "chk4_decision_grpo_steps24_sft_checkpoint_pilot_v1_20260811/"
        "sample_manifest.json",
        "1894525fa76fc947b31e78226d4b96f9d556e715ef7e8deb9e24f27f9ef3f4a2",
    ),
    "v5_grpo_smoke_authorization": (
        "output/training/retrain_v2/"
        "chk4_from_hier_balanced_v5_cp24_selected_grpo_smoke_v1_20260811/"
        "receipts/authorization.json",
        "b0317120ebc4e6a8998b11dfbfe0dadf0a76108ca29a547107ffcb3fb9745a29",
    ),
    "pre2009_stratified": (
        "output/training/retrain_v2/"
        "chk4_from_chk1_cp200_sft_grpo_pre2009_balanced_lr1e5_steps39_v1_20260811/"
        "selected_sft_checkpoints/pre2009_train_stratified_manifest_v1.json",
        "99853a20f246684b0e03a90087af5dcb3ca23732bf5db06fd5433b09516611a7",
    ),
    "correction_v1_selection": (
        "output/training/retrain_v2/"
        "chk4_from_pre2009_cp38_selected_correction_sft_lr2e6_steps6_v1_20260811/"
        "static_screening/manifests/checkpoint_selection_cap1536_v1.json",
        "e234c3bdb3021b231454c46e05d6350839d9013a40e2d9dc4afca4130c0bb9c8",
    ),
    "correction_v1_blind": (
        "output/training/retrain_v2/"
        "chk4_from_pre2009_cp38_selected_correction_sft_lr2e6_steps6_v1_20260811/"
        "static_screening/manifests/locked_checkpoint_blind_confirmation_cap1536_v1.json",
        "f30b5543383736f68c82b7cd9221e1843722b16b0e3c3fc6defdf32a9f3673d1",
    ),
}

PARTITION_SALT = "chk4-correction-v2-partition-v1"
TRAIN_SALT = "chk4-correction-v2-train-v1"
SCHEDULE_SALT = "chk4-correction-v2-schedule-v1"
PARTITION_IDS: Mapping[str, tuple[str, ...]] = {
    "selection": (
        "dec-e2006fdd25f1021f2b918f01",
        "dec-661e567e81571eb9774f86fe",
        "dec-9b8503773d5ad0d118e1b8eb",
        "dec-b2b27e148171d130a09ea6d5",
        "dec-bf310b21fe82b5c0cb502f51",
        "dec-26b22d74159882e6864770f4",
    ),
    "blind": (
        "dec-8e89d44528247ac319a2afbf",
        "dec-7a8dfb4edce9d6cc39a316eb",
        "dec-e80b4a3b73db89db41f95794",
        "dec-cfe8cd5e89cdf2d58e4f33c6",
        "dec-612552be86b5fab9002f2be4",
        "dec-b48f42df555b84e3ccf47ad7",
    ),
    "smoke": (
        "dec-b1914990f5551d18fa78a28a",
        "dec-75d77b434823305ef5db5378",
        "dec-a1b3aa8508c5c893ef267fe3",
        "dec-de51ed8cd5d92206c363873c",
    ),
    "retention": (
        "dec-aa54ffdf045b3106220b84c4",
        "dec-974aefdccb2aa2c4c14ed652",
        "dec-761b0ff6e4a9fd1de29d71f1",
        "dec-860f054c9765d7bbe54a6dec",
        "dec-5a165503b73751ffc5d8ff63",
    ),
}
PARTITION_COMMITMENTS = {
    "selection": {
        "ordered": "d5a2706f476646874a7582fe89122d67f7b439f0625526843ecea4d988a011e1",
        "set": "b28b961df84da22c5fe9af5a18f2d3e24decbc5a73a3b12c6cc0f8b7c2eee981",
    },
    "blind": {
        "ordered": "bcdc4fa3fc57770ce0ea4bae1198b95fcfd20b4584d964c63c640652dec34c40",
        "set": "96b5be0c4ebf78bb0fecbdc2c50aab2c26274f67dc66b268cdb0ee4ca5f2a206",
    },
    "smoke": {
        "ordered": "7e9a4509a05b3b405305c8f4cdda17b1a5e20d5ca8c5bb1611f19c1c659a1cb8",
        "set": "ce91538d132940093a4b435260c4ff305fe789c0fc0c4cff89a803c31091f664",
    },
    "retention": {
        "ordered": "4ca61467fd176dc2d4915baf25582d7ca3d9a2bd6fdcb83b62e2ec0739103a9e",
        "set": "a69a6f8f81634f33806ba7a9feef87a202db0f017aeac09ed8d4f6f8d7227604",
    },
}
OVERALL_ORDERED_COMMITMENT = (
    "05b76b07ee29a48d075d286c667a5238f58000e469444ecbba8215c3a1766120"
)
OVERALL_SET_COMMITMENT = (
    "b1676665b00384dadb76f4fc29ebaf96584fe1801286ee5b1793c9c307ac8ba6"
)

UNIQUE_CELL_COUNTS = {
    "core:hold": 6,
    "supplement:hold": 4,
    "core:hike": 8,
    "supplement:hike": 8,
    "core:cut": 3,
    "supplement:cut": 4,
}
REPEATED_CELL_COUNTS = {
    "core:hold": 0,
    "supplement:hold": 2,
    "core:hike": 4,
    "supplement:hike": 4,
    "core:cut": 3,
    "supplement:cut": 2,
}
EXPECTED_UNIQUE_ACTIONS = {
    "hold:0": 10,
    "hike:25": 15,
    "hike:50": 1,
    "cut:25": 3,
    "cut:50": 2,
    "cut:75": 1,
    "cut:100": 1,
}
WINDOW_SLOTS = (
    "core:hold",
    "supplement:hold",
    "core:hike",
    "core:hike",
    "supplement:hike",
    "supplement:hike",
    "core:cut",
    "supplement:cut",
)
ROWS = 48
UNIQUE_ROWS = 33
OPTIMIZER_STEPS = 6
EFFECTIVE_BATCH_SIZE = 8


class CorrectionV2ReleaseError(RuntimeError):
    """The correction-v2 data or lineage failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorrectionV2ReleaseError(message)


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
    return legacy._sha256_file(path)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    return legacy._read_json(path, label=label)


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    return legacy._read_jsonl(path, label=label)


def _row_sha(row: Mapping[str, Any]) -> str:
    return _sha256_text(_canonical_json(dict(row)))


def _rank(sample_id: str, salt: str) -> str:
    return _sha256_text(sample_id + "|" + salt)


def _cell(row: Mapping[str, Any]) -> str:
    return f"{row.get('population_role')}:{row.get('direction')}"


def _lineage(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    """Return the four identities that must remain disjoint across domains."""
    result = {
        "sample_id": {str(row["sample_id"]) for row in rows},
        "prompt_sha256": {str(row["prompt_sha256"]) for row in rows},
        "meeting_date": {str(row["meeting_date"]) for row in rows},
        "source_ids": set(),
    }
    for row in rows:
        source_ids = row.get("source_ids")
        _require(
            isinstance(source_ids, list)
            and bool(source_ids)
            and all(isinstance(value, str) and value for value in source_ids),
            f"invalid source_ids lineage for {row.get('sample_id')}",
        )
        result["source_ids"].update(source_ids)
    return result


def _lineage_receipt(lineage: Mapping[str, set[str]]) -> dict[str, Any]:
    return {
        key: {
            "count": len(values),
            "set_sha256": _sha256_text(_canonical_json(sorted(values))),
        }
        for key, values in lineage.items()
    }


def _runtime_contract(contracts: Mapping[str, Any]) -> dict[str, Any]:
    """Bind materialization and replay to the exact tokenizer audit runtime."""
    import inspect

    import accelerate
    import datasets
    import peft
    import tokenizers
    import tokenizers.tokenizers as tokenizers_native
    import torch
    import transformers
    import trl
    from transformers import AutoTokenizer

    executable = Path(sys.executable).absolute()
    _require(
        executable == EXPECTED_PYTHON, "correction-v2 must use the train env Python"
    )
    _require(
        sys.version_info[:3] == EXPECTED_PYTHON_VERSION,
        "correction-v2 Python version drift",
    )
    modules = {
        "accelerate": accelerate,
        "datasets": datasets,
        "peft": peft,
        "tokenizers": tokenizers,
        "torch": torch,
        "transformers": transformers,
        "trl": trl,
    }
    versions = {name: str(module.__version__) for name, module in modules.items()}
    _require(
        versions == EXPECTED_PACKAGE_VERSIONS, "correction-v2 package version drift"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        Path(str(contracts["tokenizer_root"])).resolve(), local_files_only=True
    )
    _require(
        type(tokenizer).__module__
        == "transformers.models.llama.tokenization_llama_fast"
        and type(tokenizer).__qualname__ == "LlamaTokenizerFast",
        "correction-v2 resolved tokenizer implementation drift",
    )
    implementation_paths = {
        "python_executable": executable.resolve(),
        "transformers_auto_tokenizer": Path(inspect.getfile(AutoTokenizer)).resolve(),
        "resolved_tokenizer_class": Path(inspect.getfile(type(tokenizer))).resolve(),
        "tokenizers_python": Path(str(tokenizers.__file__)).resolve(),
        "tokenizers_native": Path(str(tokenizers_native.__file__)).resolve(),
    }
    _require(
        all(
            path.is_file() and not path.is_symlink()
            for path in implementation_paths.values()
        ),
        "correction-v2 runtime implementation path missing or unsafe",
    )
    return {
        "schema_version": "chk4-correction-v2-materialization-runtime-v1",
        "python": {
            "executable": str(executable),
            "version": ".".join(str(value) for value in EXPECTED_PYTHON_VERSION),
        },
        "package_versions": versions,
        "tokenizer_implementation": {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "files": {
                name: {"path": str(path), "sha256": _sha256_file(path)}
                for name, path in implementation_paths.items()
            },
        },
    }


def _collect_parent_sample_ids(value: Any, parent_ids: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"sample_id", "source_sample_id"} and item in parent_ids:
                found.add(str(item))
            found.update(_collect_parent_sample_ids(item, parent_ids))
    elif isinstance(value, list):
        for item in value:
            found.update(_collect_parent_sample_ids(item, parent_ids))
    elif isinstance(value, str) and value in parent_ids:
        found.add(value)
    return found


def _burned_contract(parent_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parent_ids = {str(row["sample_id"]) for row in parent_rows}
    witnesses: dict[str, Any] = {}
    observed: set[str] = set()
    for name, (relative, expected_sha) in WITNESS_ARTIFACTS.items():
        path = (REPO_ROOT / relative).resolve()
        _require(
            _sha256_file(path) == expected_sha, f"burned witness {name} hash drift"
        )
        payload = _read_json(path, label=f"burned witness {name}")
        ids = sorted(_collect_parent_sample_ids(payload, parent_ids))
        _require(ids, f"burned witness {name} has no parent sample IDs")
        observed.update(ids)
        witnesses[name] = {
            "path": str(path),
            "sha256": expected_sha,
            "parent_sample_ids": ids,
        }
    expected = set(BURNED_SAMPLE_IDS)
    _require(observed == expected, "burned sample union drift")
    canonical_set = {
        "schema_version": "chk4-correction-v2-burned-set-v1",
        "sample_ids": sorted(observed),
    }
    _require(
        _sha256_text(_canonical_json(canonical_set)) == BURNED_SET_SHA256,
        "burned set commitment drift",
    )
    return {
        **canonical_set,
        "set_sha256": BURNED_SET_SHA256,
        "training_semantics": (
            "IDs only are observed provenance; prompts/responses are read solely "
            "from the sealed parent source_unique_train"
        ),
        "witnesses": witnesses,
    }


def _v1_training_ids(
    parent_rows: Sequence[Mapping[str, Any]],
) -> tuple[set[str], dict[str, Any]]:
    _require(
        _sha256_file(V1_RELEASE / "release_manifest.json")
        == V1_RELEASE_MANIFEST_SHA256,
        "correction-v1 release manifest drift",
    )
    _require(
        _sha256_file(V1_RELEASE / "decision_sft/train.jsonl") == V1_TRAIN_SHA256,
        "correction-v1 train hash drift",
    )
    # Do not call the mutable checkout's v1 semantic verifier here. The v1
    # release is already sealed and externally pinned, so independently verify
    # its complete file inventory, handoff, and exact teacher rows instead.
    manifest = _read_json(V1_RELEASE / "release_manifest.json", label="v1 manifest")
    _require(
        manifest.get("release_id") == "chk4_decision_pre2009_correction_sft_v1_20260811"
        and manifest.get("schema_version")
        == "chk4-decision-pre2009-correction-sft-release-v1"
        and manifest.get("quality_status") == "passed"
        and manifest.get("training_ready") is True
        and manifest.get("immutable") is True,
        "correction-v1 sealed identity drift",
    )
    file_table = manifest.get("files")
    _require(isinstance(file_table, Mapping), "correction-v1 file table missing")
    members = list(V1_RELEASE.rglob("*"))
    _require(
        all(not path.is_symlink() for path in members), "correction-v1 contains symlink"
    )
    observed_files = {
        path.relative_to(V1_RELEASE).as_posix()
        for path in members
        if path.is_file()
        and path.relative_to(V1_RELEASE).as_posix()
        not in {"release_manifest.json", "handoff.json"}
    }
    _require(set(file_table) == observed_files, "correction-v1 file inventory drift")
    for relative, record in file_table.items():
        _require(
            isinstance(record, Mapping), f"correction-v1 invalid file record {relative}"
        )
        path = V1_RELEASE / relative
        _require(
            _sha256_file(path) == record.get("sha256")
            and path.stat().st_size == record.get("bytes"),
            f"correction-v1 file binding drift {relative}",
        )
    _require(
        not (V1_RELEASE.stat().st_mode & 0o222)
        and all(not (path.stat().st_mode & 0o222) for path in members),
        "correction-v1 release is not sealed",
    )
    handoff = _read_json(V1_RELEASE / "handoff.json", label="v1 handoff")
    handoff_unsigned = dict(handoff)
    handoff_manifest_sha = handoff_unsigned.pop("release_manifest_sha256", None)
    _require(
        handoff_manifest_sha == V1_RELEASE_MANIFEST_SHA256,
        "correction-v1 handoff pin drift",
    )
    manifest_handoff = manifest.get("handoff")
    _require(
        isinstance(manifest_handoff, Mapping), "correction-v1 handoff record missing"
    )
    _require(
        handoff_unsigned.get("schema_version")
        == "chk4-decision-pre2009-correction-sft-handoff-v1"
        and handoff_unsigned.get("release_id")
        == "chk4_decision_pre2009_correction_sft_v1_20260811"
        and handoff_unsigned.get("quality_status") == "passed"
        and handoff_unsigned.get("immutable") is True
        and handoff_unsigned.get("training_ready") is True,
        "correction-v1 handoff identity/readiness drift",
    )
    _require(
        manifest_handoff.get("path") == "handoff.json"
        and manifest_handoff.get("schema_version") == handoff_unsigned["schema_version"]
        and manifest_handoff.get("unsigned_payload_sha256")
        == _sha256_text(_canonical_json(handoff_unsigned)),
        "correction-v1 handoff unsigned payload drift",
    )
    rows = _read_jsonl(V1_RELEASE / "decision_sft/train.jsonl", label="v1 train")
    ids = {str(row["source_sample_id"]) for row in rows}
    parent_by_id = {str(row["sample_id"]): row for row in parent_rows}
    _require(
        len(ids) == 48 and ids <= set(parent_by_id), "correction-v1 source IDs drift"
    )
    for row in rows:
        parent = parent_by_id[str(row["source_sample_id"])]
        _require(
            row.get("prompt") == parent.get("prompt")
            and row.get("response") == parent.get("response"),
            "correction-v1 teacher row does not match sealed parent",
        )
    return ids, {
        "release_path": str(V1_RELEASE),
        "release_manifest_sha256": V1_RELEASE_MANIFEST_SHA256,
        "train_sha256": V1_TRAIN_SHA256,
        "physical_rows": 48,
        "unique_source_rows": 48,
        "verification": "external-pin-complete-inventory-handoff-and-parent-teacher-replay-v1",
        "source_sample_ids_set_sha256": _sha256_text(_canonical_json(sorted(ids))),
    }


def _partition_payload(
    name: str, ids: Sequence[str], by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    ordered = {
        "schema_version": PARTITION_SCHEMA,
        "salt": PARTITION_SALT,
        "partition": name,
        "rows": [
            {
                "slot": slot,
                "sample_id": sample_id,
                "population_role": by_id[sample_id]["population_role"],
                "direction": by_id[sample_id]["direction"],
                "magnitude_bp": by_id[sample_id]["magnitude_bp"],
                "meeting_date": by_id[sample_id]["meeting_date"],
                "prompt_sha256": by_id[sample_id]["prompt_sha256"],
                "source_ids_sha256": _sha256_text(
                    _canonical_json(by_id[sample_id]["source_ids"])
                ),
                "rank_sha256": _rank(sample_id, PARTITION_SALT),
                "source_row_sha256": _row_sha(by_id[sample_id]),
            }
            for slot, sample_id in enumerate(ids)
        ],
    }
    unordered = {
        "schema_version": PARTITION_SCHEMA,
        "salt": PARTITION_SALT,
        "partition": name,
        "sample_ids": sorted(ids),
    }
    return ordered, unordered


def _derive_partitions(
    parent_rows: Sequence[Mapping[str, Any]],
    burned: set[str],
    v1_train: set[str],
) -> dict[str, Any]:
    by_id = {str(row["sample_id"]): row for row in parent_rows}
    fresh = [
        row for row in parent_rows if str(row["sample_id"]) not in burned | v1_train
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in fresh:
        grouped[_cell(row)].append(row)
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                _rank(str(row["sample_id"]), PARTITION_SALT),
                str(row["sample_id"]),
            )
        )
    cursors: Counter[str] = Counter()
    specs = {
        "selection": (
            ("supplement:hold", 1),
            ("supplement:hike", 2),
            ("supplement:cut", 2),
            ("core:hold", 1),
        ),
        "blind": (
            ("supplement:hold", 1),
            ("supplement:hike", 2),
            ("supplement:cut", 2),
            ("core:hold", 1),
        ),
        "smoke": (
            ("supplement:hold", 1),
            ("supplement:hike", 1),
            ("supplement:cut", 1),
            ("core:hold", 1),
        ),
    }
    derived: dict[str, tuple[str, ...]] = {}
    for name, stage_specs in specs.items():
        selected: list[str] = []
        for cell, count in stage_specs:
            start = cursors[cell]
            stop = start + count
            _require(len(grouped[cell]) >= stop, f"partition pool exhausted for {cell}")
            selected.extend(str(row["sample_id"]) for row in grouped[cell][start:stop])
            cursors[cell] = stop
        derived[name] = tuple(selected)

    v1_rows = [by_id[sample_id] for sample_id in v1_train]
    base_retention: list[str] = []
    for direction in ("hike", "cut"):
        candidates = sorted(
            (
                row
                for row in v1_rows
                if row["population_role"] == "core"
                and row["direction"] == direction
                and row["sample_id"] not in burned
            ),
            key=lambda row: (
                _rank(str(row["sample_id"]), PARTITION_SALT),
                str(row["sample_id"]),
            ),
        )
        _require(candidates, f"retention pool empty for core {direction}")
        base_retention.append(str(candidates[0]["sample_id"]))
    rare_retention: list[str] = []
    for direction, magnitude in (("hike", 50), ("cut", 75)):
        candidates = sorted(
            (
                row
                for row in v1_rows
                if row["population_role"] == "supplement"
                and row["direction"] == direction
                and int(row["magnitude_bp"]) == magnitude
            ),
            key=lambda row: (
                _rank(str(row["sample_id"]), TRAIN_SALT),
                str(row["sample_id"]),
            ),
        )
        _require(
            len(candidates) >= 2,
            f"rare retention pool too small for {direction}:{magnitude}",
        )
        # The train anchor is the minimum TRAIN_SALT rank; every remaining
        # v1-seen rare row is retained, ordered by the partition salt.
        remaining = sorted(
            candidates[1:],
            key=lambda row: (
                _rank(str(row["sample_id"]), PARTITION_SALT),
                str(row["sample_id"]),
            ),
        )
        rare_retention.extend(str(row["sample_id"]) for row in remaining)
    derived["retention"] = tuple(base_retention + rare_retention)
    _require(derived == dict(PARTITION_IDS), "partition derivation drift")
    all_ids = [sample_id for ids in derived.values() for sample_id in ids]
    _require(len(all_ids) == 21 and len(set(all_ids)) == 21, "partition overlap drift")
    _require(not set(all_ids) & burned, "partition overlaps burned rows")
    lineage = {
        name: _lineage([by_id[sample_id] for sample_id in derived[name]])
        for name in ("selection", "blind", "smoke", "retention")
    }
    partition_names = tuple(lineage)
    for index, left_name in enumerate(partition_names):
        for right_name in partition_names[index + 1 :]:
            for key in ("sample_id", "prompt_sha256", "meeting_date", "source_ids"):
                _require(
                    not lineage[left_name][key] & lineage[right_name][key],
                    f"partition lineage overlap: {left_name}/{right_name}/{key}",
                )
    prior_lineage = _lineage(
        [row for row in parent_rows if str(row["sample_id"]) in burned | v1_train]
    )
    fresh_prior_overlap_counts = {
        name: {
            key: len(lineage[name][key] & prior_lineage[key])
            for key in ("sample_id", "prompt_sha256", "meeting_date", "source_ids")
        }
        for name in ("selection", "blind", "smoke")
    }
    _require(
        all(
            count == 0
            for receipt in fresh_prior_overlap_counts.values()
            for count in receipt.values()
        ),
        "fresh partition lineage overlaps burned/correction-v1 training lineage",
    )
    _require(
        set(derived["retention"]) <= v1_train
        and set(derived["retention"]).isdisjoint(burned),
        "retention is not an unburned correction-v1 training subset",
    )

    ordered_payloads: dict[str, dict[str, Any]] = {}
    set_payloads: dict[str, dict[str, Any]] = {}
    for name in ("selection", "blind", "smoke", "retention"):
        ordered, unordered = _partition_payload(name, derived[name], by_id)
        _require(
            _sha256_text(_canonical_json(ordered))
            == PARTITION_COMMITMENTS[name]["ordered"],
            f"{name} ordered commitment drift",
        )
        _require(
            _sha256_text(_canonical_json(unordered))
            == PARTITION_COMMITMENTS[name]["set"],
            f"{name} set commitment drift",
        )
        ordered_payloads[name] = ordered
        set_payloads[name] = unordered
    overall_ordered = {
        "schema_version": PARTITION_SCHEMA,
        "salt": PARTITION_SALT,
        "partitions": [
            ordered_payloads[name]
            for name in ("selection", "blind", "smoke", "retention")
        ],
    }
    overall_set = {
        "schema_version": PARTITION_SCHEMA,
        "salt": PARTITION_SALT,
        "partition_sets": {
            name: sorted(derived[name])
            for name in ("selection", "blind", "smoke", "retention")
        },
    }
    _require(
        _sha256_text(_canonical_json(overall_ordered)) == OVERALL_ORDERED_COMMITMENT,
        "overall ordered partition commitment drift",
    )
    _require(
        _sha256_text(_canonical_json(overall_set)) == OVERALL_SET_COMMITMENT,
        "overall set partition commitment drift",
    )
    return {
        "ids": derived,
        "ordered_payloads": ordered_payloads,
        "set_payloads": set_payloads,
        "lineage": lineage,
        "lineage_receipts": {
            name: _lineage_receipt(lineage[name]) for name in partition_names
        },
        "prior_training_lineage_receipt": _lineage_receipt(prior_lineage),
        "fresh_prior_overlap_counts": fresh_prior_overlap_counts,
        "retention_v1_source_sample_ids_set_sha256": _sha256_text(
            _canonical_json(sorted(derived["retention"]))
        ),
        "overall_ordered_sha256": OVERALL_ORDERED_COMMITMENT,
        "overall_set_sha256": OVERALL_SET_COMMITMENT,
        "fresh_pool_rows": len(fresh),
    }


def _select_training_rows(
    parent_rows: Sequence[Mapping[str, Any]],
    burned: set[str],
    excluded_lineage: Mapping[str, set[str]],
    v1_train: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(row["sample_id"]): dict(row) for row in parent_rows}
    selected_ids = set(burned)
    metadata: dict[str, dict[str, Any]] = {}
    for sample_id in sorted(burned):
        metadata[sample_id] = {
            "selection_kind": "mandatory_burned",
            "train_rank_sha256": _rank(sample_id, TRAIN_SALT),
            "previously_in_correction_v1_train": sample_id in v1_train,
        }
    for cell, target in UNIQUE_CELL_COUNTS.items():
        have = sum(_cell(by_id[sample_id]) == cell for sample_id in selected_ids)
        candidates = sorted(
            (
                row
                for row in parent_rows
                if _cell(row) == cell
                and str(row["sample_id"]) not in selected_ids
                and all(
                    value not in excluded_lineage[key]
                    for key, value in (
                        ("sample_id", str(row["sample_id"])),
                        ("prompt_sha256", str(row["prompt_sha256"])),
                        ("meeting_date", str(row["meeting_date"])),
                    )
                )
                and set(row["source_ids"]).isdisjoint(excluded_lineage["source_ids"])
            ),
            key=lambda row: (
                _rank(str(row["sample_id"]), TRAIN_SALT),
                str(row["sample_id"]),
            ),
        )
        need = target - have
        _require(
            need >= 0 and len(candidates) >= need, f"training pool exhausted for {cell}"
        )
        for row in candidates[:need]:
            sample_id = str(row["sample_id"])
            selected_ids.add(sample_id)
            metadata[sample_id] = {
                "selection_kind": "deterministic_filler",
                "train_rank_sha256": _rank(sample_id, TRAIN_SALT),
                "previously_in_correction_v1_train": sample_id in v1_train,
            }
    _require(len(selected_ids) == UNIQUE_ROWS, "training unique row count drift")
    selected = sorted(
        (by_id[sample_id] for sample_id in selected_ids),
        key=lambda row: (
            _cell(row),
            _rank(str(row["sample_id"]), TRAIN_SALT),
            str(row["sample_id"]),
        ),
    )
    selected_lineage = _lineage(selected)
    for key in ("sample_id", "prompt_sha256", "meeting_date", "source_ids"):
        _require(
            not selected_lineage[key] & excluded_lineage[key],
            f"training lineage overlaps reserved partitions: {key}",
        )
    unique_cells = Counter(_cell(row) for row in selected)
    _require(dict(unique_cells) == UNIQUE_CELL_COUNTS, "training unique cell drift")
    actions = Counter(f"{row['direction']}:{row['magnitude_bp']}" for row in selected)
    _require(
        dict(actions) == EXPECTED_UNIQUE_ACTIONS, "training action magnitude drift"
    )
    meta_rows = [
        {
            "schema_version": SELECTION_SCHEMA,
            "sample_id": str(row["sample_id"]),
            "population_role": row["population_role"],
            "direction": row["direction"],
            "magnitude_bp": row["magnitude_bp"],
            "meeting_date": row["meeting_date"],
            "prompt_sha256": row["prompt_sha256"],
            "response_sha256": row["response_sha256"],
            "gold_sha256": row["gold_sha256"],
            "source_row_sha256": _row_sha(row),
            **metadata[str(row["sample_id"])],
        }
        for row in selected
    ]
    return selected, meta_rows


def _repeat_ids(selected: Sequence[Mapping[str, Any]], burned: set[str]) -> set[str]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[_cell(row)].append(row)
    result: set[str] = set()
    policies: Mapping[str, Any] = {
        "core:hold": lambda row: False,
        "supplement:hold": lambda row: row["sample_id"] not in burned,
        "core:hike": lambda row: row["sample_id"] not in burned,
        "supplement:hike": lambda row: (
            row["sample_id"] not in burned and int(row["magnitude_bp"]) == 25
        ),
        "core:cut": lambda row: True,
        "supplement:cut": lambda row: (
            row["sample_id"] not in burned and int(row["magnitude_bp"]) in {25, 50}
        ),
    }
    for cell, count in REPEATED_CELL_COUNTS.items():
        candidates = sorted(
            (row for row in grouped[cell] if policies[cell](row)),
            key=lambda row: (
                _rank(str(row["sample_id"]), TRAIN_SALT),
                str(row["sample_id"]),
            ),
        )
        _require(len(candidates) >= count, f"repeat pool exhausted for {cell}")
        result.update(str(row["sample_id"]) for row in candidates[:count])
    _require(len(result) == 15, "repeated source count drift")
    by_id = {str(row["sample_id"]): row for row in selected}
    repeated_cells = Counter(_cell(by_id[sample_id]) for sample_id in result)
    _require(
        {cell: repeated_cells.get(cell, 0) for cell in REPEATED_CELL_COUNTS}
        == REPEATED_CELL_COUNTS,
        "repeated cell drift",
    )
    _require(
        "dec-0b019571df89f22f5a5f340d" not in result
        and "dec-ea3ee432f8283bc3787c83d4" not in result,
        "supplement rare action was repeated",
    )
    _require(
        "dec-016060167a29d2b3bfa3e081" in result,
        "structurally forced core cut:100 repeat missing",
    )
    return result


def _schedule_rows(
    selected: Sequence[Mapping[str, Any]], repeated: set[str]
) -> list[dict[str, Any]]:
    by_id = {str(row["sample_id"]): row for row in selected}
    occurrences: dict[str, list[str]] = {}
    for cell in UNIQUE_CELL_COUNTS:
        unique = sorted(
            (sample_id for sample_id, row in by_id.items() if _cell(row) == cell),
            key=lambda sample_id: (_rank(sample_id, SCHEDULE_SALT), sample_id),
        )
        repeats = sorted(
            (sample_id for sample_id in unique if sample_id in repeated),
            key=lambda sample_id: (_rank(sample_id, SCHEDULE_SALT), sample_id),
        )
        occurrences[cell] = unique + repeats
        expected = 12 if cell.endswith(":hike") else 6
        _require(
            len(occurrences[cell]) == expected, f"schedule cell exposure drift: {cell}"
        )

    result: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    for step in range(OPTIMIZER_STEPS):
        cell_offsets: Counter[str] = Counter()
        for slot, cell in enumerate(WINDOW_SLOTS):
            offset = step * (2 if cell.endswith(":hike") else 1) + cell_offsets[cell]
            cell_offsets[cell] += 1
            sample_id = occurrences[cell][offset]
            source = by_id[sample_id]
            repeat_index = seen[sample_id]
            repeat_total = 2 if sample_id in repeated else 1
            seen[sample_id] += 1
            schedule_index = len(result)
            training_row_id = (
                "row-"
                + _sha256_text(
                    _canonical_json(
                        {
                            "release_id": RELEASE_ID,
                            "schedule_index": schedule_index,
                            "source_sample_id": sample_id,
                            "source_repeat_index": repeat_index,
                        }
                    )
                )[:24]
            )
            result.append(
                {
                    "schema_version": SCHEDULE_SCHEMA,
                    "schedule_index": schedule_index,
                    "optimizer_step": step,
                    "microbatch_slot": slot,
                    "training_row_id": training_row_id,
                    "source_sample_id": sample_id,
                    "source_repeat_index": repeat_index,
                    "source_repeat_total": repeat_total,
                    "population_role": source["population_role"],
                    "direction": source["direction"],
                    "magnitude_bp": source["magnitude_bp"],
                    "prompt_sha256": source["prompt_sha256"],
                    "response_sha256": source["response_sha256"],
                    "gold_sha256": source["gold_sha256"],
                }
            )
        window = result[-8:]
        _require(
            len({row["source_sample_id"] for row in window}) == 8,
            f"window {step} repeats a source",
        )
    _require(len(result) == ROWS, "schedule row count drift")
    _require(
        Counter(seen.values()) == Counter({1: 18, 2: 15}),
        "schedule repeat histogram drift",
    )
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
            "optimizer_step": row["optimizer_step"],
            "microbatch_slot": row["microbatch_slot"],
            "source_sample_id": row["source_sample_id"],
            "source_repeat_index": row["source_repeat_index"],
            "source_repeat_total": row["source_repeat_total"],
            "population_role": row["population_role"],
            "direction": row["direction"],
            "training_row_id": row["training_row_id"],
        }
        for row in schedule
    ]


def _partition_manifest_rows(
    partitions: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    return {
        name: [dict(row) for row in partitions["ordered_payloads"][name]["rows"]]
        for name in ("selection", "blind", "smoke", "retention")
    }


def _audit(
    *,
    selected: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    selection_meta: Sequence[Mapping[str, Any]],
    partitions: Mapping[str, Any],
    burned: set[str],
    contracts: Mapping[str, Any],
) -> dict[str, Any]:
    physical_ids = [str(row["source_sample_id"]) for row in schedule]
    repeated = Counter(physical_ids)
    by_id = {str(row["sample_id"]): row for row in selected}
    windows = []
    for step in range(OPTIMIZER_STEPS):
        rows = schedule[step * 8 : (step + 1) * 8]
        windows.append(
            {
                "optimizer_step": step,
                "source_sample_ids": [row["source_sample_id"] for row in rows],
                "cells": dict(
                    Counter(
                        f"{row['population_role']}:{row['direction']}" for row in rows
                    )
                ),
                "duplicate_sources": 8 - len({row["source_sample_id"] for row in rows}),
            }
        )
    actions = Counter(f"{row['direction']}:{row['magnitude_bp']}" for row in selected)
    physical_actions = Counter(
        f"{by_id[sample_id]['direction']}:{by_id[sample_id]['magnitude_bp']}"
        for sample_id in physical_ids
    )
    training_lineage = _lineage(selected)
    overlap_counts = {
        name: {
            key: len(training_lineage[key] & partitions["lineage"][name][key])
            for key in ("sample_id", "prompt_sha256", "meeting_date", "source_ids")
        }
        for name in ("selection", "blind", "smoke", "retention")
    }
    _require(
        all(
            count == 0
            for receipt in overlap_counts.values()
            for count in receipt.values()
        ),
        "training lineage overlaps a reserved partition",
    )
    return {
        "schema_version": AUDIT_SCHEMA,
        "quality_status": "passed",
        "unique_rows": len(selected),
        "physical_rows": len(schedule),
        "burned_unique_rows": len(burned),
        "burned_physical_rows": sum(repeated[sample_id] for sample_id in burned),
        "deterministic_filler_rows": sum(
            row["selection_kind"] == "deterministic_filler" for row in selection_meta
        ),
        "unique_cell_counts": dict(Counter(_cell(row) for row in selected)),
        "physical_cell_counts": dict(
            Counter(f"{row['population_role']}:{row['direction']}" for row in schedule)
        ),
        "unique_action_counts": dict(actions),
        "physical_action_counts": dict(physical_actions),
        "repeat_histogram": {
            str(key): value for key, value in sorted(Counter(repeated.values()).items())
        },
        "repeated_source_ids": sorted(
            sample_id for sample_id, count in repeated.items() if count == 2
        ),
        "rare_exposure": {
            "supplement_hike_50": repeated["dec-0b019571df89f22f5a5f340d"],
            "supplement_cut_75": repeated["dec-ea3ee432f8283bc3787c83d4"],
            "core_cut_100": repeated["dec-016060167a29d2b3bfa3e081"],
            "core_cut_100_repeat_is_structurally_forced": True,
        },
        "partition_commitments": {
            name: dict(PARTITION_COMMITMENTS[name])
            for name in ("selection", "blind", "smoke", "retention")
        }
        | {
            "overall_ordered": partitions["overall_ordered_sha256"],
            "overall_set": partitions["overall_set_sha256"],
        },
        "partition_lineage_receipts": partitions["lineage_receipts"],
        "prior_training_lineage_receipt": partitions["prior_training_lineage_receipt"],
        "fresh_partition_prior_overlap_counts": partitions[
            "fresh_prior_overlap_counts"
        ],
        "retention_v1_source_sample_ids_set_sha256": partitions[
            "retention_v1_source_sample_ids_set_sha256"
        ],
        "training_lineage_receipt": _lineage_receipt(training_lineage),
        "training_partition_overlap_counts": overlap_counts,
        "windows": windows,
        "token_budget": legacy._token_budget(selected, schedule, contracts),
    }


def _file_records(root: Path) -> dict[str, dict[str, Any]]:
    return legacy._file_records(root)


def _manifest_payload(
    *,
    root: Path,
    parent: Path,
    parent_model: Mapping[str, Any],
    contracts: Mapping[str, Any],
    burned_contract: Mapping[str, Any],
    v1_contract: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    partitions: Mapping[str, Any],
    audit: Mapping[str, Any],
    created_at: str,
    handoff_unsigned: Mapping[str, Any],
) -> dict[str, Any]:
    files = _file_records(root)
    return {
        "schema_version": RELEASE_SCHEMA,
        "release_id": RELEASE_ID,
        "release_type": "burned_plus_fresh_direction_correction_sft_train_only",
        "dataset_role": DATASET_ROLE,
        "created_at_utc": created_at,
        "quality_status": "passed",
        "training_ready": True,
        "immutable": True,
        "canonical_dag_bindable": False,
        "test_is_sealed_evaluation_only": True,
        "parent_release": {
            "path": str(parent),
            "release_id": PARENT_RELEASE_ID,
            "manifest_sha256": PARENT_MANIFEST_SHA256,
        },
        "training_parent_model": dict(parent_model),
        "materialization_runtime": dict(runtime_contract),
        "correction_v1_training": dict(v1_contract),
        "burned_contract": dict(burned_contract),
        "partition_contract": {
            "schema_version": PARTITION_SCHEMA,
            "salt": PARTITION_SALT,
            "fresh_pool_rows": partitions["fresh_pool_rows"],
            "partitions": {
                name: {
                    "rows": len(PARTITION_IDS[name]),
                    "ordered_sha256": PARTITION_COMMITMENTS[name]["ordered"],
                    "set_sha256": PARTITION_COMMITMENTS[name]["set"],
                    "path": f"manifests/partitions/{name}.jsonl",
                }
                for name in ("selection", "blind", "smoke", "retention")
            },
            "overall_ordered_sha256": OVERALL_ORDERED_COMMITMENT,
            "overall_set_sha256": OVERALL_SET_COMMITMENT,
            "partition_lineage_receipts": partitions["lineage_receipts"],
            "fresh_partition_prior_overlap_counts": partitions[
                "fresh_prior_overlap_counts"
            ],
            "retention_contract": {
                "source": "sealed_correction_v1_train",
                "unburned": True,
                "source_sample_ids_set_sha256": partitions[
                    "retention_v1_source_sample_ids_set_sha256"
                ],
            },
            "train_exclusion_keys": [
                "sample_id",
                "prompt_sha256",
                "meeting_date",
                "source_ids",
            ],
            "future_generation_contract": {
                "stage_domain_separated_seeds_required": True,
                "first_bound_eos_raw_and_normalized_audit_required": True,
                "selection_cannot_read_blind_or_retention_results": True,
            },
        },
        "selection_contract": {
            "algorithm": "mandatory-burned-plus-cell-hash-ranked-filler-v1",
            "train_salt": TRAIN_SALT,
            "unique_source_rows": UNIQUE_ROWS,
            "unique_cell_counts": UNIQUE_CELL_COUNTS,
            "parent_teacher_rows_only": True,
            "historical_completion_text_forbidden": True,
            "source_selection_path": "manifests/source_selection.jsonl",
            "source_selection_meta_path": "manifests/source_selection_meta.jsonl",
        },
        "unique_split_counts": {"train": UNIQUE_ROWS, "validation": 13, "test": 13},
        "physical_split_counts": {
            "decision_sft": {"train": ROWS, "validation": 13, "test": 13}
        },
        "files": files,
        "sampler_contract": {
            "type": SAMPLER_TYPE,
            "schedule_path": "manifests/sampler_schedule.jsonl",
            "schedule_sha256": files["manifests/sampler_schedule.jsonl"]["sha256"],
            "schedule_rows": ROWS,
            "train_path": "decision_sft/train.jsonl",
            "train_sha256": files["decision_sft/train.jsonl"]["sha256"],
            "train_rows": ROWS,
            "unique_source_rows": UNIQUE_ROWS,
            "optimizer_steps": OPTIMIZER_STEPS,
            "effective_batch_size": 8,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "world_size": 1,
            "shuffle_dataset": False,
            "direction_counts": {"hold": 12, "hike": 24, "cut": 12},
            "per_optimizer_window": {"hold": 2, "hike": 4, "cut": 2},
            "population_counts": {"core": 24, "supplement": 24},
            "per_optimizer_window_population": {"core": 4, "supplement": 4},
            "per_optimizer_window_cells": {
                "core:hold": 1,
                "supplement:hold": 1,
                "core:hike": 2,
                "supplement:hike": 2,
                "core:cut": 1,
                "supplement:cut": 1,
            },
            "unique_source_cells": UNIQUE_CELL_COUNTS,
            "repeated_source_cells": REPEATED_CELL_COUNTS,
            "order_is_authoritative": True,
            "optimizer_windows_with_duplicate_source": 0,
            "source_population": "pre2009_correction_v2_burned_plus_fresh",
            "max_source_repeat": 2,
            "repeat_histogram": {"1": 18, "2": 15},
            "required_sampler": "fixed_sequential_schedule_index_correction_v2",
            "secondary_shuffle_forbidden": True,
        },
        "training_role": {
            "dataset_path": "decision_sft",
            "completion_only_loss": True,
            "max_length": 3072,
            "parent_role": "selected_pre2009_cp38_exact_merged",
            "sampler_contract": "sampler_contract",
        },
        "evaluation_inheritance": {
            split: {
                "source_path": str(parent / f"decision_sft/{split}.jsonl"),
                "sha256": _sha256_file(parent / f"decision_sft/{split}.jsonl"),
                "rows": len(
                    _read_jsonl(
                        parent / f"decision_sft/{split}.jsonl", label=f"parent {split}"
                    )
                ),
                "copied_byte_for_byte": True,
            }
            for split in ("validation", "test")
        },
        "input_contract": {
            "path": "contracts/decision_input_contract.json",
            "schema_version": contracts["input_contract"]["schema_version"],
            "contract_sha256": contracts["input_contract"]["contract_sha256"],
            "file_sha256": files["contracts/decision_input_contract.json"]["sha256"],
            "inherited_byte_for_byte": True,
        },
        "token_contract": {
            **dict(contracts["token_contract"]),
            "tokenizer_bundle_sha256": contracts["tokenizer_bundle_sha256"],
            "parent_release_manifest_sha256": PARENT_MANIFEST_SHA256,
        },
        "audit": {
            "path": "audits/data_quality.json",
            "sha256": files["audits/data_quality.json"]["sha256"],
            "quality_status": "passed",
        },
        "implementation": {
            "materializer_snapshot": "provenance/materializer_snapshot.py",
            "materializer_snapshot_sha256": files[
                "provenance/materializer_snapshot.py"
            ]["sha256"],
            "legacy_helper_snapshot": "provenance/legacy_helper_snapshot.py",
            "legacy_helper_snapshot_sha256": files[
                "provenance/legacy_helper_snapshot.py"
            ]["sha256"],
        },
        "limitations": {
            "burned_rows_are_not_evaluation_rows": True,
            "clean_fresh_core_action_pool_rows": 0,
            "core_action_fillers_reuse_correction_v1_train": True,
            "core_cut_100_is_repeated_by_schedule_structure": True,
            "pre2009_has_no_independent_validation_or_test": True,
        },
        "handoff": {
            "path": "handoff.json",
            "schema_version": HANDOFF_SCHEMA,
            "unsigned_payload_sha256": _sha256_text(_canonical_json(handoff_unsigned)),
        },
    }


def _derive_all(parent: Path) -> dict[str, Any]:
    verified_parent = legacy._verify_parent(parent)
    contracts = legacy._parent_contracts(parent, verified_parent)
    runtime_contract = _runtime_contract(contracts)
    parent_rows = legacy._parent_source_rows(parent)
    _require(len(parent_rows) == 211, "sealed parent unique train row count drift")
    by_id = {str(row["sample_id"]): row for row in parent_rows}
    _require(len(by_id) == 211, "sealed parent sample IDs are not unique")
    burned_contract = _burned_contract(parent_rows)
    burned = set(BURNED_SAMPLE_IDS)
    v1_ids, v1_contract = _v1_training_ids(parent_rows)
    partitions = _derive_partitions(parent_rows, burned, v1_ids)
    excluded_lineage = {
        key: set().union(
            *(partitions["lineage"][name][key] for name in partitions["lineage"])
        )
        for key in ("sample_id", "prompt_sha256", "meeting_date", "source_ids")
    }
    selected, selection_meta = _select_training_rows(
        parent_rows, burned, excluded_lineage, v1_ids
    )
    repeated = _repeat_ids(selected, burned)
    schedule = _schedule_rows(selected, repeated)
    train = _sft_rows(schedule, selected)
    parent_model = fingerprint_artifact_path(TRAINING_PARENT_MODEL)
    _require(
        parent_model["sha256"] == TRAINING_PARENT_SHA256,
        "exact cp38 parent fingerprint drift",
    )
    audit = _audit(
        selected=selected,
        schedule=schedule,
        selection_meta=selection_meta,
        partitions=partitions,
        burned=burned,
        contracts=contracts,
    )
    return {
        "verified_parent": verified_parent,
        "contracts": contracts,
        "runtime_contract": runtime_contract,
        "parent_rows": parent_rows,
        "by_id": by_id,
        "burned_contract": burned_contract,
        "v1_contract": v1_contract,
        "v1_train_ids": v1_ids,
        "partitions": partitions,
        "selected": selected,
        "selection_meta": selection_meta,
        "repeated": repeated,
        "schedule": schedule,
        "train": train,
        "parent_model": parent_model,
        "audit": audit,
    }


def build_release(parent: Path, staging: Path) -> dict[str, Any]:
    derived = _derive_all(parent)
    legacy._write_jsonl(
        staging / "manifests/source_selection.jsonl", derived["selected"]
    )
    legacy._write_jsonl(
        staging / "manifests/source_selection_meta.jsonl", derived["selection_meta"]
    )
    for name, rows in _partition_manifest_rows(derived["partitions"]).items():
        legacy._write_jsonl(staging / f"manifests/partitions/{name}.jsonl", rows)
    legacy._write_jsonl(
        staging / "manifests/sampler_schedule.jsonl", derived["schedule"]
    )
    legacy._write_jsonl(staging / "decision_sft/train.jsonl", derived["train"])
    for split in ("validation", "test"):
        destination = staging / f"decision_sft/{split}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / f"decision_sft/{split}.jsonl", destination)
    input_destination = staging / "contracts/decision_input_contract.json"
    input_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(derived["contracts"]["input_path"], input_destination)
    legacy._write_json(staging / "audits/data_quality.json", derived["audit"])
    snapshot = staging / "provenance/materializer_snapshot.py"
    helper_snapshot = staging / "provenance/legacy_helper_snapshot.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).resolve(), snapshot)
    shutil.copyfile(Path(legacy.__file__).resolve(), helper_snapshot)
    created_at = legacy._utc_now()
    handoff_unsigned = {
        "schema_version": HANDOFF_SCHEMA,
        "release_id": RELEASE_ID,
        "created_at_utc": created_at,
        "dataset_role": DATASET_ROLE,
        "quality_status": "passed",
        "immutable": True,
        "training_ready": True,
        "release_manifest": "release_manifest.json",
        "dataset_path": "decision_sft",
        "physical_train_rows": ROWS,
        "unique_train_rows": UNIQUE_ROWS,
        "optimizer_steps": OPTIMIZER_STEPS,
        "test_is_sealed_evaluation_only": True,
    }
    manifest = _manifest_payload(
        root=staging,
        parent=parent,
        parent_model=derived["parent_model"],
        contracts=derived["contracts"],
        burned_contract=derived["burned_contract"],
        v1_contract=derived["v1_contract"],
        runtime_contract=derived["runtime_contract"],
        partitions=derived["partitions"],
        audit=derived["audit"],
        created_at=created_at,
        handoff_unsigned=handoff_unsigned,
    )
    legacy._write_json(staging / "release_manifest.json", manifest)
    manifest_sha = _sha256_file(staging / "release_manifest.json")
    legacy._write_json(
        staging / "handoff.json",
        {**handoff_unsigned, "release_manifest_sha256": manifest_sha},
    )
    return manifest


def verify_release(
    root: Path,
    *,
    expected_manifest_sha256: str,
    require_release_name: bool = True,
    require_sealed: bool = True,
) -> dict[str, Any]:
    release = root.expanduser().resolve()
    _require(release.is_dir() and not root.is_symlink(), "invalid correction-v2 root")
    if require_release_name:
        _require(release.name == RELEASE_ID, "correction-v2 release directory drift")
    manifest_path = release / "release_manifest.json"
    observed_sha = _sha256_file(manifest_path)
    _require(
        len(expected_manifest_sha256) == 64
        and observed_sha == expected_manifest_sha256,
        "correction-v2 external manifest pin mismatch",
    )
    manifest = _read_json(manifest_path, label="correction-v2 manifest")
    _require(
        manifest.get("schema_version") == RELEASE_SCHEMA
        and manifest.get("release_id") == RELEASE_ID
        and manifest.get("dataset_role") == DATASET_ROLE
        and manifest.get("quality_status") == "passed"
        and manifest.get("training_ready") is True
        and manifest.get("immutable") is True,
        "correction-v2 identity/readiness drift",
    )
    parent_record = manifest.get("parent_release")
    _require(isinstance(parent_record, Mapping), "correction-v2 parent binding missing")
    _require(
        parent_record.get("release_id") == PARENT_RELEASE_ID
        and parent_record.get("manifest_sha256") == PARENT_MANIFEST_SHA256,
        "correction-v2 parent binding drift",
    )
    parent = legacy._resolve_parent(str(parent_record.get("path")))
    derived = _derive_all(parent)

    files = manifest.get("files")
    _require(isinstance(files, Mapping), "correction-v2 file inventory missing")
    members = list(release.rglob("*"))
    _require(
        all(not path.is_symlink() for path in members), "correction-v2 contains symlink"
    )
    observed_files = {
        path.relative_to(release).as_posix()
        for path in members
        if path.is_file()
        and path.relative_to(release).as_posix()
        not in {"release_manifest.json", "handoff.json"}
    }
    _require(set(files) == observed_files, "correction-v2 file inventory drift")
    for name, record in files.items():
        _require(isinstance(record, Mapping), f"invalid file record {name}")
        path = release / name
        _require(path.is_file() and not path.is_symlink(), f"missing file {name}")
        _require(_sha256_file(path) == record.get("sha256"), f"file hash drift {name}")
        _require(path.stat().st_size == record.get("bytes"), f"file size drift {name}")
        if name.endswith(".jsonl"):
            _require(
                len(_read_jsonl(path, label=name)) == record.get("rows"),
                f"row count drift {name}",
            )
    if require_sealed:
        _require(not (release.stat().st_mode & 0o222), "correction-v2 root is writable")
        _require(
            all(not (path.stat().st_mode & 0o222) for path in members),
            "correction-v2 member is writable",
        )

    _require(
        (release / "provenance/materializer_snapshot.py").read_bytes()
        == Path(__file__).resolve().read_bytes(),
        "correction-v2 materializer snapshot drift",
    )
    _require(
        (release / "provenance/legacy_helper_snapshot.py").read_bytes()
        == Path(legacy.__file__).resolve().read_bytes(),
        "correction-v2 legacy helper snapshot drift",
    )
    _require(
        _read_jsonl(
            release / "manifests/source_selection.jsonl", label="source selection"
        )
        == derived["selected"],
        "source selection replay drift",
    )
    _require(
        _read_jsonl(
            release / "manifests/source_selection_meta.jsonl",
            label="source selection meta",
        )
        == derived["selection_meta"],
        "source selection metadata replay drift",
    )
    for name, rows in _partition_manifest_rows(derived["partitions"]).items():
        _require(
            _read_jsonl(release / f"manifests/partitions/{name}.jsonl", label=name)
            == rows,
            f"partition {name} replay drift",
        )
    _require(
        _read_jsonl(release / "manifests/sampler_schedule.jsonl", label="schedule")
        == derived["schedule"],
        "schedule replay drift",
    )
    _require(
        _read_jsonl(release / "decision_sft/train.jsonl", label="SFT train")
        == derived["train"],
        "SFT train replay drift",
    )
    for split in ("validation", "test"):
        _require(
            (release / f"decision_sft/{split}.jsonl").read_bytes()
            == (parent / f"decision_sft/{split}.jsonl").read_bytes(),
            f"{split} byte inheritance drift",
        )
    _require(
        (release / "contracts/decision_input_contract.json").read_bytes()
        == Path(derived["contracts"]["input_path"]).read_bytes(),
        "input contract byte inheritance drift",
    )
    audit = _read_json(release / "audits/data_quality.json", label="data audit")
    _require(audit == derived["audit"], "data audit replay drift")

    handoff = _read_json(release / "handoff.json", label="handoff")
    unsigned = dict(handoff)
    handoff_manifest_sha = unsigned.pop("release_manifest_sha256", None)
    _require(handoff_manifest_sha == observed_sha, "handoff manifest pin drift")
    manifest_handoff = manifest.get("handoff")
    _require(isinstance(manifest_handoff, Mapping), "handoff manifest record missing")
    _require(
        unsigned.get("schema_version") == HANDOFF_SCHEMA
        and unsigned.get("release_id") == RELEASE_ID
        and unsigned.get("dataset_role") == DATASET_ROLE
        and unsigned.get("quality_status") == "passed"
        and unsigned.get("immutable") is True
        and unsigned.get("training_ready") is True,
        "handoff identity/readiness drift",
    )
    expected_unsigned = {
        "schema_version": HANDOFF_SCHEMA,
        "release_id": RELEASE_ID,
        "created_at_utc": manifest.get("created_at_utc"),
        "dataset_role": DATASET_ROLE,
        "quality_status": "passed",
        "immutable": True,
        "training_ready": True,
        "release_manifest": "release_manifest.json",
        "dataset_path": "decision_sft",
        "physical_train_rows": ROWS,
        "unique_train_rows": UNIQUE_ROWS,
        "optimizer_steps": OPTIMIZER_STEPS,
        "test_is_sealed_evaluation_only": True,
    }
    _require(unsigned == expected_unsigned, "handoff semantic payload drift")
    _require(
        manifest_handoff.get("path") == "handoff.json"
        and manifest_handoff.get("schema_version") == HANDOFF_SCHEMA
        and manifest_handoff.get("unsigned_payload_sha256")
        == _sha256_text(_canonical_json(unsigned)),
        "handoff unsigned payload binding drift",
    )
    expected_manifest = _manifest_payload(
        root=release,
        parent=parent,
        parent_model=derived["parent_model"],
        contracts=derived["contracts"],
        burned_contract=derived["burned_contract"],
        v1_contract=derived["v1_contract"],
        runtime_contract=derived["runtime_contract"],
        partitions=derived["partitions"],
        audit=derived["audit"],
        created_at=str(manifest.get("created_at_utc")),
        handoff_unsigned=unsigned,
    )
    _require(
        manifest == expected_manifest, "correction-v2 manifest semantic replay drift"
    )
    return {**manifest, "release_manifest_sha256": observed_sha}


def verify_runtime_release(
    *,
    dataset_dir: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    dataset_role: str,
    model_path: Path,
) -> dict[str, Any]:
    _require(dataset_role == DATASET_ROLE, "correction-v2 runtime role drift")
    manifest_path = manifest_path.expanduser().resolve()
    release = manifest_path.parent
    _require(
        dataset_dir.expanduser().resolve() == release / "decision_sft",
        "correction-v2 dataset path drift",
    )
    verified = verify_release(
        release, expected_manifest_sha256=expected_manifest_sha256
    )
    observed_model = fingerprint_artifact_path(model_path)
    _require(
        observed_model == verified["training_parent_model"],
        "correction-v2 runtime parent model drift",
    )
    sampler = dict(verified["sampler_contract"])
    sampler["schedule_path"] = str(release / sampler["schedule_path"])
    sampler["train_path"] = str(release / sampler["train_path"])
    return {
        "schema_version": "chk4-pre2009-correction-v2-runtime-binding-v1",
        "release_id": RELEASE_ID,
        "dataset_role": DATASET_ROLE,
        "release_manifest_path": str(manifest_path),
        "release_manifest_sha256": expected_manifest_sha256,
        "split_files": {
            "train": release / "decision_sft/train.jsonl",
            "validation": release / "decision_sft/validation.jsonl",
        },
        "test_verified_but_not_loaded": True,
        "tokenizer_binding": {
            "path": str(derived_path)
            if (
                derived_path := Path(str(verified["token_contract"]["tokenizer_path"]))
            ).is_absolute()
            else str((REPO_ROOT / derived_path).resolve()),
            "bundle_sha256": verified["token_contract"]["tokenizer_bundle_sha256"],
            "files": verified["token_contract"]["tokenizer_files"],
        },
        "sampler_contract": sampler,
        "training_parent_model": observed_model,
        "partition_contract": verified["partition_contract"],
    }


def materialize(parent: Path, output: Path) -> dict[str, Any]:
    parent = legacy._resolve_parent(parent)
    output = output.expanduser().resolve()
    _require(output.name == RELEASE_ID, "correction-v2 output directory name drift")
    _require(
        not output.exists() and not output.is_symlink(),
        "correction-v2 output already exists",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{RELEASE_ID}.", dir=output.parent))
    staging = staging_root / RELEASE_ID
    staging.mkdir()
    published_by_this_call = False
    try:
        build_release(parent, staging)
        manifest_sha = _sha256_file(staging / "release_manifest.json")
        verify_release(
            staging,
            expected_manifest_sha256=manifest_sha,
            require_release_name=True,
            require_sealed=False,
        )
        # Seal every member while the tree is still private. Keep only the
        # staging root at 0700 because renameat2 on this filesystem rejects a
        # non-writable source root. After the atomic no-clobber rename, the
        # only remaining operation is chmod of the root itself.
        for path in sorted(
            staging.rglob("*"), key=lambda item: len(item.parts), reverse=True
        ):
            path.chmod(0o555 if path.is_dir() else 0o444)
        _require(
            all(not (path.stat().st_mode & 0o222) for path in staging.rglob("*")),
            "correction-v2 staging members were not sealed",
        )
        staging.chmod(0o700)
        legacy._rename_noreplace(staging, output)
        published_by_this_call = True
        output.chmod(0o555)
        verified = verify_release(output, expected_manifest_sha256=manifest_sha)
        return verified
    finally:
        if published_by_this_call and output.exists() and not output.is_symlink():
            # Idempotent safety net for an interruption immediately after the
            # successful atomic rename.
            output.chmod(0o555)
        if staging_root.exists():
            _require(
                staging_root.parent == output.parent
                and staging_root.name.startswith(f".{RELEASE_ID}.")
                and not staging_root.is_symlink(),
                f"refusing unsafe correction-v2 staging cleanup: {staging_root}",
            )
            cleanup_members = list(staging_root.rglob("*"))
            _require(
                all(not path.is_symlink() for path in cleanup_members),
                "refusing correction-v2 staging cleanup containing a symlink",
            )
            for path in cleanup_members:
                path.chmod(0o700 if path.is_dir() else 0o600)
            staging_root.chmod(0o700)
            shutil.rmtree(staging_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    publish.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--release", type=Path, required=True)
    verify.add_argument("--expected-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "publish":
        result = materialize(args.parent, args.output)
    else:
        result = verify_release(
            args.release,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
