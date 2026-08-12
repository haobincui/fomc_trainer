"""Pre-register and run correction-v2 static generation panels.

This module is additive: it does not modify the sealed correction-v1 probes or
receipts.  It freezes four disjoint panels before correction-v2 SFT:

* ``selection`` compares exact cp38 with checkpoints 2, 4, and 6;
* ``blind`` is evaluated only for the earliest locked checkpoint;
* ``retention`` is a second, disjoint locked-checkpoint confirmation; and
* ``grpo_smoke`` reserves the two optimizer-step prompt layout used later by
  the authoritative GRPO smoke (this static probe never executes that stage).

All sampled panels use stage-domain-separated SHA-256-derived seeds.  GPU
outputs preserve the raw padded token tensor *and* the sequence truncated at
the first bound EOS (inclusive).  Completion text, metrics, and reward are
computed from the normalized IDs and every row is flushed and fsynced before
the next row is generated.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from jobs.retrain_v2 import probe_chk4_decision_grpo_stratified as reward_probe
from jobs.retrain_v2 import probe_chk4_decision_sft_generation as generation_probe
from jobs.retrain_v2 import probe_chk4_pre2009_correction as correction_v1_probe
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


REPO_ROOT = Path(__file__).resolve().parents[2]
PARENT_RELEASE = correction_v1_probe.PARENT_RELEASE
PARENT_MANIFEST_SHA256 = correction_v1_probe.PARENT_MANIFEST_SHA256
SELECTED_CP38_MODEL = correction_v1_probe.SELECTED_CP38_MODEL
SELECTED_CP38_MODEL_SHA256 = correction_v1_probe.SELECTED_CP38_MODEL_SHA256

CORRECTION_V2_RELEASE_ID = "chk4_decision_pre2009_correction_sft_v2_20260811"
CORRECTION_V2_RELEASE_SCHEMA = "chk4-decision-pre2009-correction-sft-release-v2"
CORRECTION_V2_ROLE = "decision_sft_pre2009_correction_v2"
RUN_ROOT = REPO_ROOT / (
    "output/training/retrain_v2/"
    "chk4_from_pre2009_cp38_selected_correction_sft_v2_lr2e6_steps6_20260811"
)
# Attempt 0 stopped before model loading because the launcher passed the
# authorization file-size field into the probe's deliberately narrower
# binding schema.  Keep that immutable preflight evidence in
# ``static_screening`` and use a fresh canonical root for the corrected run.
SCREEN_ROOT = RUN_ROOT / "static_screening_attempt1_auth_binding_fix"
SFT_OUTPUT = RUN_ROOT / "adapters/chk4_correction_sft"

MANIFEST_SCHEMA = "chk4-pre2009-correction-v2-generation-suite-manifest-v1"
RESULT_SCHEMA = "chk4-pre2009-correction-v2-static-screen-result-v1"
SUMMARY_SCHEMA = "chk4-pre2009-correction-v2-static-screen-summary-v1"
PARTITION_SALT = "chk4-correction-v2-partition-v1"
MAX_NEW_TOKENS = 1536
TAIL_TOKENS = 256
NUM_RETURN_SEQUENCES = 4
TEMPERATURE = 0.7
TOP_P = 0.9

SELECTION_STAGE = "selection"
BLIND_STAGE = "blind"
RETENTION_STAGE = "retention"
GRPO_SMOKE_STAGE = "grpo_smoke"
STATIC_STAGES = (SELECTION_STAGE, BLIND_STAGE, RETENTION_STAGE)
ALL_STAGES = (
    SELECTION_STAGE,
    BLIND_STAGE,
    GRPO_SMOKE_STAGE,
    RETENTION_STAGE,
)

SELECTION_IDS = (
    "dec-e2006fdd25f1021f2b918f01",
    "dec-661e567e81571eb9774f86fe",
    "dec-9b8503773d5ad0d118e1b8eb",
    "dec-b2b27e148171d130a09ea6d5",
    "dec-bf310b21fe82b5c0cb502f51",
    "dec-26b22d74159882e6864770f4",
)
BLIND_IDS = (
    "dec-8e89d44528247ac319a2afbf",
    "dec-7a8dfb4edce9d6cc39a316eb",
    "dec-e80b4a3b73db89db41f95794",
    "dec-cfe8cd5e89cdf2d58e4f33c6",
    "dec-612552be86b5fab9002f2be4",
    "dec-b48f42df555b84e3ccf47ad7",
)
SMOKE_IDS = (
    "dec-b1914990f5551d18fa78a28a",
    "dec-75d77b434823305ef5db5378",
    "dec-a1b3aa8508c5c893ef267fe3",
    "dec-de51ed8cd5d92206c363873c",
)
RETENTION_IDS = (
    "dec-aa54ffdf045b3106220b84c4",
    "dec-974aefdccb2aa2c4c14ed652",
    "dec-761b0ff6e4a9fd1de29d71f1",
    "dec-860f054c9765d7bbe54a6dec",
    "dec-5a165503b73751ffc5d8ff63",
)
PARTITIONS = {
    SELECTION_STAGE: SELECTION_IDS,
    BLIND_STAGE: BLIND_IDS,
    GRPO_SMOKE_STAGE: SMOKE_IDS,
    RETENTION_STAGE: RETENTION_IDS,
}

STAGE_SEED_SALTS = {
    SELECTION_STAGE: "chk4-correction-v2-selection-generation-v1",
    BLIND_STAGE: "chk4-correction-v2-blind-generation-v1",
    RETENTION_STAGE: "chk4-correction-v2-retention-generation-v1",
    GRPO_SMOKE_STAGE: "chk4-correction-v2-grpo-smoke-generation-v1",
}
EXPECTED_STAGE_SEEDS = {
    SELECTION_STAGE: 12908838,
    BLIND_STAGE: 25434928,
    RETENTION_STAGE: 1163146095,
    GRPO_SMOKE_STAGE: 1772719529,
}

STAGE_BATCHES = {
    SELECTION_STAGE: (
        (SELECTION_IDS[0], SELECTION_IDS[1]),
        (SELECTION_IDS[2], SELECTION_IDS[3]),
        (SELECTION_IDS[4], SELECTION_IDS[5]),
    ),
    BLIND_STAGE: (
        (BLIND_IDS[0], BLIND_IDS[1]),
        (BLIND_IDS[2], BLIND_IDS[3]),
        (BLIND_IDS[4], BLIND_IDS[5]),
    ),
    RETENTION_STAGE: (
        (RETENTION_IDS[0], RETENTION_IDS[1]),
        (RETENTION_IDS[2], RETENTION_IDS[3]),
        (RETENTION_IDS[4],),
    ),
    # Exact optimizer-step order frozen for the later GRPO smoke.
    GRPO_SMOKE_STAGE: (
        (SMOKE_IDS[3], SMOKE_IDS[1]),  # core hold + pre-2009 hike
        (SMOKE_IDS[0], SMOKE_IDS[2]),  # pre-2009 hold + pre-2009 cut
    ),
}

CANDIDATE_ORDER = (2, 4, 6)
AUTH_IMPLEMENTATION_RELATIVES = (
    "jobs/retrain_v2/run_chk4_pre2009_correction_v2_screen.py",
    "jobs/retrain_v2/probe_chk4_pre2009_correction_v2.py",
    "jobs/retrain_v2/audit_chk4_pre2009_correction_v2.py",
    "jobs/retrain_v2/materialize_chk4_pre2009_correction_v2_release.py",
    "jobs/retrain_v2/probe_chk4_pre2009_correction.py",
    "jobs/retrain_v2/probe_chk4_decision_sft_generation.py",
    "jobs/retrain_v2/probe_chk4_decision_grpo_stratified.py",
    "src/open_r1/trainer/rewards/reward_funcs/decision_reward_v3.py",
    "src/open_r1/structured_response.py",
    "src/open_r1/provenance.py",
)


class CorrectionV2ProbeError(RuntimeError):
    """The correction-v2 generation contract failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorrectionV2ProbeError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def derive_stage_seed(stage: str) -> dict[str, Any]:
    _require(stage in ALL_STAGES, f"unsupported stage: {stage}")
    salt = STAGE_SEED_SALTS[stage]
    digest = sha256_text(salt)
    seed = int(digest[:8], 16) & 0x7FFFFFFF
    _require(seed == EXPECTED_STAGE_SEEDS[stage], f"{stage} seed drift")
    return {
        "algorithm": "int(sha256(stage_salt)[:8],16)&0x7fffffff",
        "stage_salt": salt,
        "stage_salt_sha256": digest,
        "seed": seed,
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorrectionV2ProbeError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CorrectionV2ProbeError(f"cannot read {label}: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        _require(bool(line), f"{label}:{line_number}: blank row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorrectionV2ProbeError(
                f"{label}:{line_number}: invalid JSON"
            ) from exc
        _require(isinstance(row, dict), f"{label}:{line_number}: not an object")
        rows.append(row)
    return rows


def _write_exclusive_json(
    path: Path, value: Mapping[str, Any], *, mode: int = 0o444
) -> None:
    _require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _partition_commitment() -> dict[str, Any]:
    flattened = [sample_id for stage in ALL_STAGES for sample_id in PARTITIONS[stage]]
    _require(len(flattened) == 21 and len(set(flattened)) == 21, "partitions overlap")
    per_stage: dict[str, Any] = {}
    for stage in ALL_STAGES:
        payload = {
            "partition_salt": PARTITION_SALT,
            "stage": stage,
            "sample_ids": list(PARTITIONS[stage]),
        }
        per_stage[stage] = {
            **payload,
            "payload_sha256": sha256_text(canonical_json(payload)),
        }
    overall_payload = {
        "partition_salt": PARTITION_SALT,
        "ordered_stages": list(ALL_STAGES),
        "partitions": {stage: list(PARTITIONS[stage]) for stage in ALL_STAGES},
    }
    return {
        "algorithm": "sha256(canonical-json({partition_salt,stage,sample_ids}))",
        "per_stage": per_stage,
        "overall_payload_sha256": sha256_text(canonical_json(overall_payload)),
    }


def _absolute_prompt_gate() -> dict[str, Any]:
    return {
        "hold_correct_nonzero_min": 2,
        "action_correct_nonzero_min": 1,
        "strict_json_exact_min": 1,
        "reward_variation": "pstdev_gt_0_or_strict_json_exact_eq_4",
        "boundary_count_min": 3,
        "cap_count_max": 1,
        "periodic_tail_count_max": 0,
        "all_rewards_finite": True,
    }


def _stage_contract(stage: str) -> dict[str, Any]:
    seed = derive_stage_seed(stage)
    common = {
        "sample_ids": list(PARTITIONS[stage]),
        "batches": [list(batch) for batch in STAGE_BATCHES[stage]],
        "seed": seed,
        "seed_once_before_all_batches": True,
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_return_sequences": NUM_RETURN_SEQUENCES,
            "do_sample": True,
            "raw_padded_ids_required": True,
            "first_bound_eos_inclusive_ids_required": True,
            "completion_and_reward_row_fsync_required": True,
        },
    }
    if stage in {SELECTION_STAGE, BLIND_STAGE}:
        common["quality_gate"] = {
            "per_prompt": _absolute_prompt_gate(),
            "aggregate_candidate_vs_exact_cp38_baseline": {
                "hold_correct_nonzero_min": "max(4,baseline_hold-1)",
                "hike_correct_nonzero_min": "min(8,max(2,baseline_hike+1))",
                "cut_correct_nonzero_min": "max(2,baseline_cut)",
            },
            "selection_rule": (
                "earliest_all_pass_in_preregistered_order_2_4_6"
                if stage == SELECTION_STAGE
                else "locked_checkpoint_only_failure_is_terminal"
            ),
        }
    elif stage == RETENTION_STAGE:
        common["quality_gate"] = {
            "per_prompt_candidate_vs_exact_cp38_baseline": {
                "correct_nonzero_min": "max(1,baseline_correct_nonzero)",
                "strict_json_exact_min": "baseline_strict_json_exact",
                "reward_variation": "pstdev_gt_0_or_strict_json_exact_eq_4",
                "boundary_count_min": 3,
                "cap_count_max": 1,
                "periodic_tail_count_max": 0,
                "all_rewards_finite": True,
            },
            "selection_rule": "locked_checkpoint_only_failure_is_terminal",
        }
    else:
        common["quality_gate"] = {
            "per_prompt": _absolute_prompt_gate(),
            "per_optimizer_step_delivery": {
                "boundary_count_min": 6,
                "cap_count_max": 2,
                "periodic_tail_count_max": 0,
                "all_rewards_finite": True,
            },
            "optimizer": {
                "loss_finite": True,
                "grad_norm_min_when_either_prompt_not_all_strict_exact": 1e-12,
            },
            "persistence": (
                "raw_and_normalized_ids_and_completion_before_deferred_nonzero_exit"
            ),
            "failure_is_terminal_and_blocks_full": True,
        }
    return common


def _verify_correction_v2_release(path: Path, expected_sha256: str) -> dict[str, Any]:
    root = path.expanduser().resolve().parent
    _require(root.name == CORRECTION_V2_RELEASE_ID, "correction-v2 release ID drift")
    _require(sha256_file(path) == expected_sha256, "correction-v2 manifest hash drift")
    manifest = _read_json(path, label="correction-v2 release manifest")
    _require(
        manifest.get("schema_version") == CORRECTION_V2_RELEASE_SCHEMA,
        "correction-v2 release schema drift",
    )
    # The materializer is intentionally imported lazily because preparation is
    # allowed only after the separately reviewed release is published.
    try:
        from jobs.retrain_v2.materialize_chk4_pre2009_correction_v2_release import (
            OVERALL_ORDERED_COMMITMENT,
            OVERALL_SET_COMMITMENT,
            PARTITION_COMMITMENTS,
            PARTITION_IDS,
            verify_release,
        )
    except ImportError as exc:
        raise CorrectionV2ProbeError(
            "correction-v2 materializer/verifier is not available"
        ) from exc
    try:
        verified = verify_release(root, expected_manifest_sha256=expected_sha256)
    except Exception as exc:
        raise CorrectionV2ProbeError(
            f"correction-v2 release replay failed: {exc}"
        ) from exc
    _require(isinstance(verified, Mapping), "correction-v2 verifier result invalid")
    release_expected = {
        "selection": SELECTION_IDS,
        "blind": BLIND_IDS,
        "smoke": SMOKE_IDS,
        "retention": RETENTION_IDS,
    }
    _require(dict(PARTITION_IDS) == release_expected, "publisher partition IDs drift")
    contract = verified.get("partition_contract")
    _require(isinstance(contract, Mapping), "release partition contract missing")
    release_partitions = contract.get("partitions")
    _require(
        isinstance(release_partitions, Mapping), "release partition records missing"
    )
    partition_files: dict[str, Any] = {}
    for name, sample_ids in release_expected.items():
        record = release_partitions.get(name)
        _require(isinstance(record, Mapping), f"release partition missing: {name}")
        _require(
            record.get("ordered_sha256") == PARTITION_COMMITMENTS[name]["ordered"]
            and record.get("set_sha256") == PARTITION_COMMITMENTS[name]["set"]
            and record.get("rows") == len(sample_ids),
            f"release partition commitment drift: {name}",
        )
        partition_path = root / str(record.get("path"))
        rows = _read_jsonl(partition_path, label=f"release {name} partition")
        _require(
            tuple(str(row.get("sample_id")) for row in rows) == sample_ids,
            f"release partition ID/order drift: {name}",
        )
        partition_files[name] = {
            "path": str(partition_path.resolve()),
            "sha256": sha256_file(partition_path),
            "rows": len(rows),
            "ordered_commitment_sha256": PARTITION_COMMITMENTS[name]["ordered"],
            "set_commitment_sha256": PARTITION_COMMITMENTS[name]["set"],
        }
    _require(
        contract.get("overall_ordered_sha256") == OVERALL_ORDERED_COMMITMENT
        and contract.get("overall_set_sha256") == OVERALL_SET_COMMITMENT,
        "overall release partition commitment drift",
    )
    return {
        "path": str(path.resolve()),
        "sha256": expected_sha256,
        "schema_version": CORRECTION_V2_RELEASE_SCHEMA,
        "role": CORRECTION_V2_ROLE,
        "partition_contract": _partition_commitment(),
        "source_release_partition_files": partition_files,
        "source_release_overall_ordered_sha256": OVERALL_ORDERED_COMMITMENT,
        "source_release_overall_set_sha256": OVERALL_SET_COMMITMENT,
    }


def _source_records() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    source_path = PARENT_RELEASE / "manifests/source_unique_train.jsonl"
    physical_path = PARENT_RELEASE / "decision_grpo/train.jsonl"
    parent_manifest_path = PARENT_RELEASE / "release_manifest.json"
    _require(
        sha256_file(parent_manifest_path) == PARENT_MANIFEST_SHA256,
        "parent release manifest drift",
    )
    parent_manifest = _read_json(parent_manifest_path, label="parent release manifest")
    files = parent_manifest.get("files")
    _require(isinstance(files, Mapping), "parent file table missing")
    for relative, path in (
        ("manifests/source_unique_train.jsonl", source_path),
        ("decision_grpo/train.jsonl", physical_path),
    ):
        record = files.get(relative)
        _require(isinstance(record, Mapping), f"parent file record missing: {relative}")
        _require(
            sha256_file(path) == record.get("sha256"), f"parent hash drift: {relative}"
        )

    source_rows = _read_jsonl(source_path, label="parent unique source train")
    by_id = {str(row.get("sample_id")): row for row in source_rows}
    _require(len(by_id) == len(source_rows), "parent source IDs duplicated")
    physical_rows = _read_jsonl(physical_path, label="parent GRPO train")
    prompts: dict[str, str] = {}
    lines: dict[str, list[int]] = {}
    for line_number, row in enumerate(physical_rows, 1):
        sample_id = str(row.get("sample_id"))
        prompt = str(row.get("prompt"))
        if sample_id in prompts:
            _require(prompts[sample_id] == prompt, f"prompt variants: {sample_id}")
        prompts[sample_id] = prompt
        lines.setdefault(sample_id, []).append(line_number)
    wanted = set(item for ids in PARTITIONS.values() for item in ids)
    _require(
        wanted <= set(by_id) and wanted <= set(prompts), "partition source missing"
    )
    selected: dict[str, dict[str, Any]] = {}
    for sample_id in wanted:
        source = by_id[sample_id]
        _require(
            sha256_text(prompts[sample_id]) == source.get("prompt_sha256"),
            f"prompt hash drift: {sample_id}",
        )
        selected[sample_id] = {
            **source,
            "prompt": prompts[sample_id],
            "physical_line_numbers": lines[sample_id],
        }
    return (
        {
            "unique_source": {
                "path": str(source_path.resolve()),
                "sha256": sha256_file(source_path),
                "rows": len(source_rows),
            },
            "physical_train": {
                "path": str(physical_path.resolve()),
                "sha256": sha256_file(physical_path),
                "rows": len(physical_rows),
            },
        },
        selected,
    )


def _sample_descriptor(source: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    prompt_ids = generation_probe._prompt_ids(
        tokenizer,
        generation_probe._messages(
            correction_v1_probe.CHK4_STUDENT_SYSTEM_PROMPT, str(source["prompt"])
        ),
    )
    return {
        "sample_id": source["sample_id"],
        "population_role": source["population_role"],
        "direction": source["direction"],
        "magnitude_bp": source["magnitude_bp"],
        "meeting_date": source["meeting_date"],
        "prompt_sha256": source["prompt_sha256"],
        "prompt_token_count": len(prompt_ids),
        "physical_line_numbers": source["physical_line_numbers"],
    }


def build_manifest(
    *,
    correction_release_manifest: Path,
    correction_release_manifest_sha256: str,
    tokenizer: Any,
) -> dict[str, Any]:
    """Build the deterministic, text-free four-stage suite manifest."""

    correction_v1_probe._verify_parent_release(SELECTED_CP38_MODEL)
    correction = _verify_correction_v2_release(
        correction_release_manifest.resolve(), correction_release_manifest_sha256
    )
    source_binding, sources = _source_records()
    tokenizer_binding = correction_v1_probe._tokenizer_binding()
    samples = {
        sample_id: _sample_descriptor(source, tokenizer)
        for sample_id, source in sources.items()
    }
    _require(
        max(int(row["prompt_token_count"]) for row in samples.values()) <= 2560,
        "frozen probe prompt exceeds max prompt length",
    )
    return {
        "schema_version": MANIFEST_SCHEMA,
        "methodology": {
            "name": "correction_v2_static_screening_and_locked_confirmation",
            "static_generation_not_bitwise_grpo_replay": True,
            "candidate_order": list(CANDIDATE_ORDER),
            "checkpoint_selection": "earliest_all_pass",
            "blind_and_retention_only_after_lock": True,
            "blind_or_retention_failure_terminal": True,
            "no_seed_gate_or_candidate_retry": True,
            "authoritative_grpo_gate": "fresh_two_step_cap1536_smoke",
        },
        "parent_release": {
            "path": str(PARENT_RELEASE.resolve()),
            "manifest_sha256": PARENT_MANIFEST_SHA256,
            **source_binding,
        },
        "correction_release": correction,
        "exact_cp38_baseline": fingerprint_artifact_path(SELECTED_CP38_MODEL),
        "tokenizer": tokenizer_binding,
        "partition_commitment": _partition_commitment(),
        "samples": samples,
        "stages": {stage: _stage_contract(stage) for stage in ALL_STAGES},
    }


def validate_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(sha256_file(path) == expected_sha256, "suite manifest hash drift")
    observed = _read_json(path, label="suite manifest")
    _require(observed.get("schema_version") == MANIFEST_SCHEMA, "suite schema drift")
    tokenizer_path = Path(str(observed.get("tokenizer", {}).get("path")))
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=False
        )
    except Exception as exc:
        raise CorrectionV2ProbeError(f"cannot reload bound tokenizer: {exc}") from exc
    correction = observed.get("correction_release")
    _require(isinstance(correction, Mapping), "correction release binding missing")
    rebuilt = build_manifest(
        correction_release_manifest=Path(str(correction.get("path"))),
        correction_release_manifest_sha256=str(correction.get("sha256")),
        tokenizer=tokenizer,
    )
    _require(canonical_json(rebuilt) == canonical_json(observed), "suite replay drift")
    return observed


def normalize_at_first_bound_eos(
    raw_ids: Sequence[int], eos_token_ids: set[int]
) -> tuple[list[int], dict[str, Any]]:
    _require(bool(eos_token_ids), "EOS token set is empty")
    try:
        raw = [int(value) for value in raw_ids]
    except (TypeError, ValueError) as exc:
        raise CorrectionV2ProbeError("generated IDs contain a non-integer") from exc
    first_index = next(
        (index for index, value in enumerate(raw) if value in eos_token_ids), None
    )
    normalized = raw if first_index is None else raw[: first_index + 1]
    discarded = [] if first_index is None else raw[first_index + 1 :]
    evidence = {
        "schema_version": "first-bound-eos-inclusive-normalization-v1",
        "method": "truncate_at_first_bound_eos_inclusive",
        "raw_count": len(raw),
        "first_bound_eos_index": first_index,
        "normalized_count": len(normalized),
        "discarded_count": len(discarded),
        "discarded_unique_ids": sorted(set(discarded)),
        "discarded_all_bound_eos": all(value in eos_token_ids for value in discarded),
        "hit_bound_eos": first_index is not None,
    }
    _require(
        evidence["discarded_all_bound_eos"],
        "non-EOS token appeared after first bound EOS in batched output",
    )
    return normalized, evidence


def _load_prompts(manifest: Mapping[str, Any], stage: str) -> dict[str, str]:
    stage_contract = manifest["stages"][stage]
    wanted = set(stage_contract["sample_ids"])
    rows = _read_jsonl(
        Path(str(manifest["parent_release"]["physical_train"]["path"])),
        label="bound parent GRPO train",
    )
    prompts: dict[str, str] = {}
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if sample_id not in wanted:
            continue
        prompt = str(row.get("prompt"))
        if sample_id in prompts:
            _require(prompts[sample_id] == prompt, f"prompt variants: {sample_id}")
        prompts[sample_id] = prompt
    _require(set(prompts) == wanted, "stage prompt population incomplete")
    for sample_id, prompt in prompts.items():
        _require(
            sha256_text(prompt) == manifest["samples"][sample_id]["prompt_sha256"],
            f"stage prompt hash drift: {sample_id}",
        )
    return prompts


def _model_source(
    *,
    role: str,
    model_path: Path | None,
    base_model_path: Path | None,
    adapter_path: Path | None,
) -> Mapping[str, Any]:
    _require(role in {"baseline", "candidate"}, "unsupported model role")
    if role == "baseline":
        _require(
            model_path == SELECTED_CP38_MODEL
            and base_model_path is None
            and adapter_path is None,
            "baseline model path is not canonical exact cp38",
        )
    else:
        allowed_adapters = {
            SFT_OUTPUT / f"checkpoint-{step}" for step in CANDIDATE_ORDER
        }
        _require(
            model_path is None
            and base_model_path == SELECTED_CP38_MODEL
            and adapter_path in allowed_adapters,
            "candidate base/adapter path is not a canonical correction-v2 checkpoint",
        )
    try:
        source = reward_probe._prepare_model_source(
            model_path=model_path,
            base_model_path=base_model_path,
            adapter_path=adapter_path,
        )
    except Exception as exc:
        raise CorrectionV2ProbeError(f"model source validation failed: {exc}") from exc
    if role == "baseline":
        _require(
            source["mode"] == reward_probe.MERGED_MODEL_MODE, "baseline must be merged"
        )
        _require(
            source["model_fingerprint"]["sha256"] == SELECTED_CP38_MODEL_SHA256
            and source["effective_model_fingerprint"]["sha256"]
            == SELECTED_CP38_MODEL_SHA256,
            "baseline is not exact cp38",
        )
        return {
            **source,
            "provenance": {
                **source["provenance"],
                "model_path": str(SELECTED_CP38_MODEL),
            },
        }
    else:
        _require(
            source["mode"] == reward_probe.PEFT_ADAPTER_MODE, "candidate needs adapter"
        )
        _require(
            source["model_fingerprint"]["sha256"] == SELECTED_CP38_MODEL_SHA256,
            "candidate base is not exact cp38",
        )
    return source


def _metric_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(rows) == NUM_RETURN_SEQUENCES, "prompt must have four rows")
    rewards = [float(row["decision_dense_v3_reward"]) for row in rows]
    _require(all(math.isfinite(value) for value in rewards), "non-finite reward")
    return {
        "cases": len(rows),
        "correct_nonzero_count": sum(
            bool(row["decision_exact"]) and float(row["decision_dense_v3_reward"]) > 0
            for row in rows
        ),
        "strict_json_exact_count": sum(
            bool(row["strict_json"]) and bool(row["decision_exact"]) for row in rows
        ),
        "reward_mean": round(statistics.fmean(rewards), 8),
        "reward_std": round(statistics.pstdev(rewards), 8),
        "boundary_count": sum(int(row["think_boundary_count"]) == 1 for row in rows),
        "cap_count": sum(bool(row["cap_reached"]) for row in rows),
        "periodic_tail_count": sum(bool(row["strict_periodic_tail"]) for row in rows),
        "rewards_finite": True,
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], stage: str
) -> dict[str, Any]:
    expected_ids = manifest["stages"][stage]["sample_ids"]
    _require(
        len(rows) == len(expected_ids) * NUM_RETURN_SEQUENCES, "result row count drift"
    )
    prompts: dict[str, Any] = {}
    for sample_id in expected_ids:
        sample_rows = [row for row in rows if row.get("sample_id") == sample_id]
        prompts[sample_id] = {
            "target_direction": manifest["samples"][sample_id]["direction"],
            "target_magnitude_bp": manifest["samples"][sample_id]["magnitude_bp"],
            **_metric_block(sample_rows),
        }
    aggregate = {
        direction: sum(
            block["correct_nonzero_count"]
            for block in prompts.values()
            if block["target_direction"] == direction
        )
        for direction in ("hold", "hike", "cut")
    }
    return {"prompts": prompts, "aggregate_correct_nonzero": aggregate}


def _result_row(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    stage: str,
    role: str,
    model_provenance: Mapping[str, Any],
    eos_token_ids: set[int],
    sample_id: str,
    batch_index: int,
    generation_index: int,
    raw_ids: Sequence[int],
    tokenizer: Any,
) -> dict[str, Any]:
    normalized_ids, normalization = normalize_at_first_bound_eos(raw_ids, eos_token_ids)
    completion = generation_probe.decode_completion_preserving_boundary(
        tokenizer, normalized_ids, eos_token_ids
    )
    metrics = generation_probe.analyze_completion(
        text=completion,
        generated_token_ids=normalized_ids,
        eos_token_ids=eos_token_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        tail_tokens=TAIL_TOKENS,
    )
    target = manifest["samples"][sample_id]
    replay = reward_probe.replay_decision_dense_v3(
        completion,
        {"direction": target["direction"], "magnitude_bp": target["magnitude_bp"]},
        hit_eos=bool(metrics["hit_eos"]),
        cap_reached=bool(metrics["cap_reached"]),
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "stage": stage,
        "model_role": role,
        "sample_id": sample_id,
        "population_role": target["population_role"],
        "target_direction": target["direction"],
        "target_magnitude_bp": target["magnitude_bp"],
        "batch_index": batch_index,
        "generation_index": generation_index,
        "seed": manifest["stages"][stage]["seed"]["seed"],
        "generation_parameters": manifest["stages"][stage]["generation"],
        "generated_token_ids_raw_padded": list(raw_ids),
        "generated_token_ids_first_eos_inclusive": normalized_ids,
        "batched_padding_normalization": normalization,
        "completion": completion,
        "completion_sha256": sha256_text(completion),
        **metrics,
        "decision_dense_v3_reward": replay["reward"],
        "decision_prediction": replay["prediction"],
        "decision_direction_correct": replay["direction_correct"],
        "decision_exact": replay["exact"],
        "decision_rejection_reason": replay["rejection_reason"],
        "decision_forced_zero_reason": replay["forced_zero_reason"],
        "response_format": replay["response_format"],
        "strict_json": replay["strict_json"],
        "fenced_json": replay["fenced_json"],
        "provenance": {
            "manifest_sha256": manifest_sha256,
            "authorization_sha256": model_provenance["authorization"]["sha256"],
            "base_model_sha256": model_provenance["model_sha256"],
            "effective_model_sha256": model_provenance["effective_model_sha256"],
        },
    }


def _verify_authorization_payload(
    *,
    authorization: Mapping[str, Any],
    authorization_path: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    stage: str,
    role: str,
    checkpoint_step: int | None,
    output_dir: Path,
    source: Mapping[str, Any],
) -> None:
    suffix = "baseline" if role == "baseline" else f"checkpoint-{checkpoint_step}"
    output_name = (
        "baseline-exact-cp38" if role == "baseline" else f"checkpoint-{checkpoint_step}"
    )
    expected_output = (SCREEN_ROOT / stage / output_name).resolve()
    expected_authorization = (
        SCREEN_ROOT / "authorizations" / f"{stage}-{suffix}.json"
    ).resolve()
    _require(
        output_dir.resolve() == expected_output, "screen output path is not canonical"
    )
    _require(
        authorization_path.resolve() == expected_authorization,
        "screen authorization path is not canonical",
    )
    scope = authorization.get("scope")
    _require(isinstance(scope, Mapping), "screen authorization scope missing")
    _require(
        scope.get("stage") == stage
        and scope.get("model_role") == role
        and scope.get("checkpoint_step") == checkpoint_step
        and scope.get("gpu") == 1
        and scope.get("fresh_output") is True
        and authorization.get("output") == str(output_dir.resolve()),
        "screen authorization scope drift",
    )
    _require(
        authorization.get("manifest", {}).get("path") == str(manifest_path.resolve())
        and authorization.get("manifest", {}).get("sha256") == manifest_sha256
        and authorization.get("stage_contract") == manifest["stages"][stage],
        "screen authorization manifest/stage drift",
    )
    _require(
        authorization.get("model_source") == source["provenance"]
        and authorization.get("model_sha256") == source["model_fingerprint"]["sha256"]
        and authorization.get("effective_model_sha256")
        == source["effective_model_fingerprint"]["sha256"],
        "screen authorization model fingerprint drift",
    )
    implementation = authorization.get("implementation")
    _require(
        isinstance(implementation, Mapping)
        and set(implementation) == set(AUTH_IMPLEMENTATION_RELATIVES),
        "authorization implementation inventory drift",
    )
    for relative, descriptor in implementation.items():
        _require(isinstance(descriptor, Mapping), "invalid implementation descriptor")
        implementation_path = Path(str(descriptor.get("path")))
        _require(
            implementation_path == (REPO_ROOT / relative).resolve()
            and implementation_path.is_file()
            and not implementation_path.is_symlink()
            and sha256_file(implementation_path) == descriptor.get("sha256"),
            f"authorized implementation drift: {relative}",
        )
    predecessors = authorization.get("predecessors")
    _require(
        isinstance(predecessors, Mapping)
        and set(predecessors) == {"selection", "blind"},
        "authorization predecessors missing",
    )
    needs_selection = not (
        stage == SELECTION_STAGE
        and (role == "baseline" or checkpoint_step == CANDIDATE_ORDER[0])
    )
    needs_blind = stage == RETENTION_STAGE
    _require(
        (predecessors.get("selection") is not None) == needs_selection
        and (predecessors.get("blind") is not None) == needs_blind,
        "authorization predecessor state drift",
    )
    from jobs.retrain_v2 import audit_chk4_pre2009_correction_v2 as semantic_auditor

    for name in ("selection", "blind"):
        binding = predecessors.get(name)
        if binding is None:
            continue
        _require(isinstance(binding, Mapping), f"invalid predecessor binding: {name}")
        predecessor_path = Path(str(binding.get("path")))
        predecessor = _read_json(predecessor_path, label=f"{name} predecessor")
        _require(
            predecessor_path.stat().st_mode & 0o777 == 0o400,
            f"authorization predecessor mode drift: {name}",
        )
        validate_manifest_integrity(predecessor)
        _require(
            sha256_file(predecessor_path) == binding.get("sha256")
            and predecessor.get("integrity", {}).get("payload_sha256")
            == binding.get("payload_sha256")
            and predecessor.get("status") == binding.get("status"),
            f"authorization predecessor drift: {name}",
        )
        if name == "selection":
            semantic_auditor.verify_selection_receipt(
                path=predecessor_path,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
            )
        else:
            semantic_auditor.verify_confirmation_receipt(
                path=predecessor_path,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
            )


def run_panel(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    stage: str,
    role: str,
    output_dir: Path,
    model_label: str,
    model_path: Path | None = None,
    base_model_path: Path | None = None,
    adapter_path: Path | None = None,
    load_in_4bit: bool = True,
    authorization_binding: Mapping[str, Any],
) -> dict[str, Any]:
    _require(stage in STATIC_STAGES, "grpo_smoke is manifest-only in this utility")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "probe requires GPU1")
    _require(not output_dir.exists() and not output_dir.is_symlink(), "output exists")
    _require(
        set(authorization_binding) == {"path", "sha256", "payload_sha256"},
        "authorization binding schema drift",
    )
    authorization_path = Path(str(authorization_binding["path"])).resolve()
    authorization = _read_json(authorization_path, label="screen authorization")
    _require(
        authorization_path.stat().st_mode & 0o777 == 0o400,
        "screen authorization mode drift",
    )
    validate_manifest_integrity(authorization)
    _require(
        authorization.get("schema_version")
        == "chk4-pre2009-correction-v2-static-screen-authorization-v1"
        and authorization.get("status") == "authorized"
        and sha256_file(authorization_path) == authorization_binding["sha256"]
        and authorization.get("integrity", {}).get("payload_sha256")
        == authorization_binding["payload_sha256"],
        "screen authorization hash drift",
    )
    manifest = validate_manifest(manifest_path.resolve(), manifest_sha256)
    tokenizer_path = Path(str(manifest["tokenizer"]["path"]))
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=False
        )
    except Exception as exc:
        raise CorrectionV2ProbeError(f"cannot load tokenizer: {exc}") from exc
    prompts = _load_prompts(manifest, stage)
    source = _model_source(
        role=role,
        model_path=model_path,
        base_model_path=base_model_path,
        adapter_path=adapter_path,
    )
    checkpoint_step = (
        int(Path(str(adapter_path)).name.removeprefix("checkpoint-"))
        if role == "candidate" and adapter_path is not None
        else None
    )
    _verify_authorization_payload(
        authorization=authorization,
        authorization_path=authorization_path,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        stage=stage,
        role=role,
        checkpoint_step=checkpoint_step,
        output_dir=output_dir,
        source=source,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    launch_path = output_dir / "launch.json"
    result_path = output_dir / "results.jsonl"
    source_provenance = source["provenance"]
    provenance: dict[str, Any] = {
        "manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_sha256},
        "correction_release_manifest_sha256": manifest["correction_release"]["sha256"],
        "model_source": source_provenance,
        "model_sha256": source["model_fingerprint"]["sha256"],
        "effective_model_sha256": source["effective_model_fingerprint"]["sha256"],
        "authorization": dict(authorization_binding),
    }
    launch = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "initializing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "model_role": role,
        "model_label": model_label,
        "runtime": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "cuda_visible_devices": "1",
        },
        "provenance": provenance,
    }
    _write_exclusive_json(launch_path, launch)
    model: Any = None
    rows: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        _require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "one visible GPU required",
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        quantization = None
        if load_in_4bit:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(
            str(source["load_path"]),
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            quantization_config=quantization,
            device_map={"": 0},
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        if source["mode"] == reward_probe.PEFT_ADAPTER_MODE:
            model = reward_probe._attach_local_peft_adapter(
                model,
                adapter_path=source["adapter_path"],
                adapter_config=source["adapter_config"],
            )
        model.eval()
        model.config.use_cache = True
        eos_value = getattr(model.generation_config, "eos_token_id", None)
        if eos_value is None:
            eos_value = tokenizer.eos_token_id
        eos_token_ids = set(generation_probe._normalize_eos_ids(eos_value))
        _require(bool(eos_token_ids), "model/tokenizer has no EOS")
        provenance["eos_token_ids"] = sorted(eos_token_ids)
        torch.cuda.reset_peak_memory_stats()
        transformers.set_seed(int(manifest["stages"][stage]["seed"]["seed"]))

        def render(sample_id: str) -> str:
            return tokenizer.apply_chat_template(
                generation_probe._messages(
                    correction_v1_probe.CHK4_STUDENT_SYSTEM_PROMPT, prompts[sample_id]
                ),
                tokenize=False,
                add_generation_prompt=True,
            )

        def persist(handle: IO[str], row: Mapping[str, Any]) -> None:
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        with result_path.open("x", encoding="utf-8") as handle:
            for batch_index, sample_ids in enumerate(
                manifest["stages"][stage]["batches"]
            ):
                encoded = tokenizer(
                    [render(str(sample_id)) for sample_id in sample_ids],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                input_ids = encoded["input_ids"].to("cuda:0")
                attention_mask = encoded["attention_mask"].to("cuda:0")
                with torch.inference_mode():
                    sequences = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        num_return_sequences=NUM_RETURN_SEQUENCES,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=sorted(eos_token_ids),
                        use_cache=True,
                    )
                _require(
                    len(sequences) == len(sample_ids) * NUM_RETURN_SEQUENCES,
                    "batched generation row count drift",
                )
                prompt_width = int(input_ids.shape[1])
                for prompt_index, sample_id in enumerate(sample_ids):
                    for generation_index in range(NUM_RETURN_SEQUENCES):
                        sequence_index = (
                            prompt_index * NUM_RETURN_SEQUENCES + generation_index
                        )
                        raw_ids = sequences[sequence_index, prompt_width:].tolist()
                        row = _result_row(
                            manifest=manifest,
                            manifest_sha256=manifest_sha256,
                            stage=stage,
                            role=role,
                            model_provenance=provenance,
                            eos_token_ids=eos_token_ids,
                            sample_id=str(sample_id),
                            batch_index=batch_index,
                            generation_index=generation_index + 1,
                            raw_ids=raw_ids,
                            tokenizer=tokenizer,
                        )
                        rows.append(row)
                        persist(handle, row)
        result_path.chmod(0o444)
        metrics = summarize_rows(rows, manifest, stage)
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "status": "generation_complete",
            "quality_status": "not_evaluated_without_paired_baseline_or_candidate",
            "stage": stage,
            "model_role": role,
            "model_label": model_label,
            "cases": len(rows),
            "metrics": metrics,
            "elapsed_seconds": round(
                (datetime.now(timezone.utc) - started).total_seconds(), 3
            ),
            "runtime": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated() / 1024**3, 3
                ),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved() / 1024**3, 3
                ),
            },
            "results": {
                "path": str(result_path.resolve()),
                "sha256": sha256_file(result_path),
                "rows": len(rows),
                "raw_padded_and_normalized_ids_landed": True,
                "row_fsync_required": True,
            },
            "provenance": provenance,
        }
        _write_exclusive_json(output_dir / "summary.json", summary)
        output_dir.chmod(0o555)
        return summary
    except Exception as exc:
        if result_path.exists():
            result_path.chmod(0o444)
        _write_exclusive_json(
            output_dir / "failure.json",
            {
                **launch,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_cases": len(rows),
            },
        )
        output_dir.chmod(0o555)
        raise
    finally:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--correction-release-manifest", type=Path, required=True)
    prepare.add_argument("--correction-release-manifest-sha256", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        tokenizer_binding = correction_v1_probe._tokenizer_binding()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_binding["path"]),
            local_files_only=True,
            trust_remote_code=False,
        )
        manifest = build_manifest(
            correction_release_manifest=args.correction_release_manifest,
            correction_release_manifest_sha256=args.correction_release_manifest_sha256,
            tokenizer=tokenizer,
        )
        _write_exclusive_json(args.output, manifest)
        result = {
            "status": "prepared",
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "stages": list(ALL_STAGES),
            "generation_execution": (
                "requires_create_only_authorization_via_"
                "run_chk4_pre2009_correction_v2_screen"
            ),
        }
        print(canonical_json(result))
        return 0
    except (CorrectionV2ProbeError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
