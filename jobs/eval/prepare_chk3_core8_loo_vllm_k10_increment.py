"""Prepare the create-only CHK3 Core8 LOO K=6--K=10 increment.

The completed K=5 release is immutable.  This profile reuses its audited
preparation mechanics, but emits a new cohort containing only five fresh
stochastic replicates.  Runtime ``replicate_id`` values are local (0--4) so
that the frozen dual-DP1 sharding contract remains unchanged; the sealed
global identities are 5--9 and are carried separately.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import prepare_chk3_core8_loo_vllm_k5 as frozen
from open_r1.validator.loo_generation_spec import seal_manifest as _seal_manifest


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ID = (
    "chk3-cp318-core8-loo-n128-vllm-k10-increment-k6-k10-dual-dp1-v1"
)

COHORT_SCHEMA = "chk3-core8-loo-vllm-k10-increment-k6-k10-cohort-v1"
LEDGER_ROW_SCHEMA = (
    "chk3-core8-loo-vllm-k10-increment-prompt-token-ledger-row-v1"
)
PREPARATION_SCHEMA = (
    "chk3-core8-loo-vllm-k10-increment-k6-k10-preparation-v1"
)
EXECUTION_POLICY_SCHEMA = (
    "chk3-core8-loo-vllm-k10-increment-k6-k10-execution-policy-v1"
)

DEFAULT_SOURCE_SAMPLE_MANIFEST = frozen.DEFAULT_SOURCE_SAMPLE_MANIFEST
SOURCE_SAMPLE_MANIFEST_SHA256 = frozen.SOURCE_SAMPLE_MANIFEST_SHA256
DEFAULT_RUN_ROOT = ROOT / (
    "output/evaluation/main/"
    "chk3_cp318_core8_loo_vllm_k10_n128_1993_2008_20260824_v1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_ROOT / "preparation_increment_k6_k10_v1"

MODEL_ID = frozen.MODEL_ID
EXPECTED_PROMPTS = frozen.EXPECTED_PROMPTS
EXPECTED_MEETINGS = frozen.EXPECTED_MEETINGS
VARIANTS_PER_MEETING = frozen.VARIANTS_PER_MEETING
REPLICATE_SEEDS = (
    25260811,
    26260811,
    27260811,
    28260811,
    29260811,
)
LOCAL_REPLICATE_IDS = tuple(range(len(REPLICATE_SEEDS)))
GLOBAL_REPLICATE_OFFSET = 5
GLOBAL_REPLICATE_IDS = tuple(
    GLOBAL_REPLICATE_OFFSET + value for value in LOCAL_REPLICATE_IDS
)
PRIOR_COMPLETED_REPLICATES = 5
INCREMENTAL_REPLICATES = len(REPLICATE_SEEDS)
COMBINED_REPLICATES = PRIOR_COMPLETED_REPLICATES + INCREMENTAL_REPLICATES
EXPECTED_CASES = EXPECTED_PROMPTS * INCREMENTAL_REPLICATES

MAX_NEW_TOKENS = frozen.MAX_NEW_TOKENS
MAX_MODEL_LEN = frozen.MAX_MODEL_LEN
MAX_PROMPT_TOKENS = frozen.MAX_PROMPT_TOKENS
ABSOLUTE_CHUNK_SIZE = VARIANTS_PER_MEETING * INCREMENTAL_REPLICATES * 2
EXPECTED_CHUNKS = EXPECTED_CASES // ABSOLUTE_CHUNK_SIZE

LEDGER_FILENAME = "prompt_token_ledger_n2176.increment-k6-k10.v1.jsonl"
COHORT_FILENAME = "cohort_n2176_increment_k6_k10.v1.json"
PREPARATION_FILENAME = "preparation.increment-k6-k10.v1.json"
EXECUTION_POLICY_FILENAME = "execution_policy.increment-k6-k10.v1.json"

PRIOR_K5_COHORT = frozen.DEFAULT_OUTPUT_DIR / frozen.COHORT_FILENAME
PRIOR_K5_COHORT_SHA256 = (
    "26c1a06231fd8b10a0ccf2a9de22a59b7d657696da88a18981bad6fb82762e1c"
)
PRIOR_K5_POLICY = frozen.DEFAULT_OUTPUT_DIR / frozen.EXECUTION_POLICY_FILENAME
PRIOR_K5_POLICY_SHA256 = (
    "611b25af74ac21409071ab10ce08574d2afe9aaf38b920d9f40e02b3febbbde3"
)
INCOMPLETE_NF4_K10_ROOT = ROOT / (
    "output/evaluation/main/"
    "chk3_cp318_core8_loo_stochastic_k10_n128_1993_2008_20260815_v1/"
    "generation_v1"
)
INCOMPLETE_NF4_K10_ARTIFACTS = {
    "launch": (
        INCOMPLETE_NF4_K10_ROOT / "launch.json",
        "760204a36af5e6b3318a51eccb329b55be79f56d94fa6c716b2a81d0e03d5aeb",
        None,
    ),
    "state": (
        INCOMPLETE_NF4_K10_ROOT / "state.progress.v1.json",
        "e74bbbbf398252299fffc28514ac3bd0445257bf9d7afe99845e7c212cdb6562",
        None,
    ),
    "partial_generations": (
        INCOMPLETE_NF4_K10_ROOT / ".partial/generations.progress.v1.jsonl",
        "2ef8a604fcbb2a57397f50841e2901fa8eca483b8d44b4a0a2a99a5d24b95ae4",
        2_071,
    ),
}


class LooVllmK10IncrementPreparationError(RuntimeError):
    """The sealed K=6--K=10 incremental preparation contract was violated."""


# Compatibility name consumed by the frozen K5 runner while it is scoped to
# this preparation module.  It remains the original exception type so Python
# can use it in an ``except`` clause.
LooVllmK5PreparationError = frozen.LooVllmK5PreparationError


def replicate_indexing_contract() -> dict[str, Any]:
    """Return the immutable mapping from local runtime IDs to final K=10 IDs."""

    return {
        "replicate_id_scope": "increment_local",
        "local_replicate_ids": list(LOCAL_REPLICATE_IDS),
        "global_replicate_offset": GLOBAL_REPLICATE_OFFSET,
        "global_replicate_ids": list(GLOBAL_REPLICATE_IDS),
        "replicate_seed_by_global_id": {
            str(global_id): seed
            for global_id, seed in zip(
                GLOBAL_REPLICATE_IDS, REPLICATE_SEEDS, strict=True
            )
        },
        "prior_completed_replicates": PRIOR_COMPLETED_REPLICATES,
        "incremental_replicates": INCREMENTAL_REPLICATES,
        "combined_replicates": COMBINED_REPLICATES,
        "prior_k5_rows_reused_in_increment": False,
    }


def _checked_binding(
    path: Path, expected_sha256: str, *, rows: int | None = None
) -> dict[str, Any]:
    binding = frozen._binding(path, rows=rows)
    if binding["sha256"] != expected_sha256:
        raise LooVllmK10IncrementPreparationError(
            f"frozen K5 artifact binding drift: {path}"
        )
    return binding


def prior_k5_bindings() -> dict[str, Any]:
    """Deep-bind the immutable five-replicate predecessor release."""

    return {
        "role": "immutable_predecessor_not_rewritten_or_reused_in_increment",
        "cohort": _checked_binding(PRIOR_K5_COHORT, PRIOR_K5_COHORT_SHA256),
        "execution_policy": _checked_binding(
            PRIOR_K5_POLICY, PRIOR_K5_POLICY_SHA256
        ),
    }


def excluded_incomplete_nf4_k10_bindings() -> dict[str, Any]:
    """Bind the abandoned NF4 partial run and explicitly exclude every row."""

    return {
        "role": "excluded_incomplete_nf4_k10_historical_evidence_only",
        "included_in_increment": False,
        "included_in_combined_k10": False,
        "reason": "incomplete_and_different_nf4_runtime_contract",
        "artifacts": {
            name: _checked_binding(path, sha256, rows=rows)
            for name, (path, sha256, rows) in INCOMPLETE_NF4_K10_ARTIFACTS.items()
        },
    }


def increment_authorization() -> dict[str, Any]:
    return {
        "status": "user_authorized",
        "authorization_date": "2026-08-24",
        "directive": "add_five_fresh_replicates_to_extend_frozen_k5_to_k10",
        "authorized_local_replicate_ids": list(LOCAL_REPLICATE_IDS),
        "authorized_global_replicate_ids": list(GLOBAL_REPLICATE_IDS),
        "reuse_or_mutation_of_prior_k5": False,
        "gpu_generation_requires_separate_smoke_authorization": True,
    }


def _implementation_bindings() -> dict[str, Any]:
    return {
        "increment_preparer_adapter": frozen._binding(Path(__file__).resolve()),
        "frozen_k5_preparer_base": frozen._binding(
            Path(frozen.__file__).resolve()
        ),
    }


def _seal_increment_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Add increment lineage before sealing; never mutate the caller's object."""

    payload = copy.deepcopy(dict(value))
    payload["replicate_indexing"] = replicate_indexing_contract()
    payload["increment_authorization"] = increment_authorization()
    payload["prior_k5_release"] = prior_k5_bindings()
    payload["excluded_incomplete_nf4_k10"] = (
        excluded_incomplete_nf4_k10_bindings()
    )
    implementation = payload.get("implementation")
    if implementation is None:
        implementation = {}
    if not isinstance(implementation, Mapping):
        raise LooVllmK10IncrementPreparationError(
            "manifest implementation field is not an object"
        )
    payload["implementation"] = {
        **dict(implementation),
        **_implementation_bindings(),
    }
    return _seal_manifest(payload)


def _frozen_replacements() -> dict[str, Any]:
    return {
        "__doc__": __doc__,
        "EVALUATION_ID": EVALUATION_ID,
        "COHORT_SCHEMA": COHORT_SCHEMA,
        "LEDGER_ROW_SCHEMA": LEDGER_ROW_SCHEMA,
        "PREPARATION_SCHEMA": PREPARATION_SCHEMA,
        "EXECUTION_POLICY_SCHEMA": EXECUTION_POLICY_SCHEMA,
        "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
        "REPLICATE_SEEDS": REPLICATE_SEEDS,
        "EXPECTED_CASES": EXPECTED_CASES,
        "ABSOLUTE_CHUNK_SIZE": ABSOLUTE_CHUNK_SIZE,
        "EXPECTED_CHUNKS": EXPECTED_CHUNKS,
        "LEDGER_FILENAME": LEDGER_FILENAME,
        "COHORT_FILENAME": COHORT_FILENAME,
        "PREPARATION_FILENAME": PREPARATION_FILENAME,
        "EXECUTION_POLICY_FILENAME": EXECUTION_POLICY_FILENAME,
        "seal_manifest": _seal_increment_manifest,
    }


@contextlib.contextmanager
def configured_frozen_preparer() -> Any:
    """Scope all K5-base substitutions and restore them even after failure."""

    replacements = _frozen_replacements()
    previous = {name: getattr(frozen, name) for name in replacements}
    for name, value in replacements.items():
        setattr(frozen, name, value)
    try:
        yield frozen
    finally:
        for name, value in previous.items():
            setattr(frozen, name, value)


def canonical_case_coordinates(absolute_case_index: int) -> dict[str, int]:
    """Return local execution and global K=10 coordinates for one case."""

    with configured_frozen_preparer():
        try:
            result = frozen.canonical_case_coordinates(absolute_case_index)
        except frozen.LooVllmK5PreparationError as exc:
            raise LooVllmK10IncrementPreparationError(str(exc)) from exc
    local_id = int(result["replicate_id"])
    return {
        **result,
        "local_replicate_id": local_id,
        "global_replicate_id": GLOBAL_REPLICATE_OFFSET + local_id,
    }


def build_execution_policy(**kwargs: Any) -> dict[str, Any]:
    with configured_frozen_preparer():
        try:
            return frozen.build_execution_policy(**kwargs)
        except frozen.LooVllmK5PreparationError as exc:
            raise LooVllmK10IncrementPreparationError(str(exc)) from exc


def prepare(
    *,
    source_sample_manifest_path: Path = DEFAULT_SOURCE_SAMPLE_MANIFEST,
    source_sample_manifest_sha256: str = SOURCE_SAMPLE_MANIFEST_SHA256,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Create the new sealed increment without touching the K=1 or K=5 roots."""

    with configured_frozen_preparer():
        try:
            return frozen.prepare(
                source_sample_manifest_path=source_sample_manifest_path,
                source_sample_manifest_sha256=source_sample_manifest_sha256,
                output_dir=output_dir,
            )
        except frozen.LooVllmK5PreparationError as exc:
            raise LooVllmK10IncrementPreparationError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-sample-manifest",
        type=Path,
        default=DEFAULT_SOURCE_SAMPLE_MANIFEST,
    )
    parser.add_argument(
        "--source-sample-manifest-sha256",
        default=SOURCE_SAMPLE_MANIFEST_SHA256,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare(
            source_sample_manifest_path=args.source_sample_manifest,
            source_sample_manifest_sha256=args.source_sample_manifest_sha256,
            output_dir=args.output_dir,
        )
    except (LooVllmK10IncrementPreparationError, OSError, ValueError) as exc:
        print(
            frozen._canonical(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        )
        return 1
    print(frozen._canonical(result))
    return 0


def __getattr__(name: str) -> Any:
    value = getattr(frozen, name)
    if isinstance(value, type):
        return value
    if callable(value):
        def scoped(*args: Any, **kwargs: Any) -> Any:
            with configured_frozen_preparer():
                return getattr(frozen, name)(*args, **kwargs)

        return scoped
    return value


if __name__ == "__main__":
    raise SystemExit(main())
