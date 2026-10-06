"""Run CHK0/CHK1/CHK3 on the merged 256-meeting Core8 panel at K=10.

This is a versioned profile over the existing durable stochastic generator.
The underlying runner retains GPU0 exclusivity, per-row RNG reset,
append/flush/fsync WAL, exact-prefix resume, model order CHK1 -> CHK3 -> CHK0,
and sealed manifests.  This profile freezes N=2,048 prompts, K=10, source
lineage metadata, and chronological meeting/topic ordering.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import chk3_beta_core8_merged_contract as contract
from jobs.eval import eval_chk3_stochastic_bootstrap_generation as core


ROOT = Path(__file__).resolve().parents[2]
REPLICATE_SEEDS = (
    20260811,
    21260811,
    22260811,
    23260811,
    24260811,
    25260811,
    26260811,
    27260811,
    28260811,
    29260811,
)
BOOTSTRAP_SEED = 20260815
PROFILE_SCHEMA = "chk3-beta-core8-merged-n2048-k10-profile-v1"
RESEARCH_SCOPE = "formal_merged_panel_descriptive"
TRANSPORT_SPLIT_ROLE = "test_compatibility_shim_not_a_held_out_claim"
PROFILE_METADATA_KEYS = (
    "era",
    "source_split",
    "original_post_split_role",
    "original_qa_split",
    "meeting_type",
    "sensitivity_flag",
    "sensitivity_reason",
    "scheduled",
    "meeting_start_date",
    "meeting_end_date",
    "evidence_cutoff",
    "topic",
    "topic_order",
    "cp318_selection_exposed",
    "source_sample_id",
    "source_id",
    "source_core8_line_number",
    "prompt_sha256",
    "source_analysis_sha256",
    "reference_minutes_sha256",
    "topic_evidence_sha256",
    "source_release",
    "research_scope",
    "transport_split_role",
    "not_all_held_out",
)

_ORIGINAL_BUILD_RESULT = core.build_stochastic_result
_ORIGINAL_VALIDATE_RESULT = core.validate_stochastic_result
_ORIGINAL_SOURCE_HASHES = core._source_hashes
_ORIGINAL_LOAD_SAMPLE_MANIFEST = core.load_full_test_sample_manifest
_ORIGINAL_SHA256_FILE = core._sha256_file
_ORIGINAL_VALIDATE_RESUME_STATE = core._validate_resume_state
_ORIGINAL_SAMPLING_CONTRACT = core._sampling_contract
_ACTIVE_SOURCES: contract.HarmonizedSources | None = None
_EXPECTED_SAMPLE_METADATA: dict[str, dict[str, Any]] = {}
_WAL_CACHE: dict[Path, dict[str, Any]] = {}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _is_generation_wal(path: Path) -> bool:
    return path.name in {"generations.progress.v1.jsonl", "generations.jsonl"}


def _scan_wal(path: Path) -> dict[str, Any]:
    """Hash a WAL once (fresh run: never; resume/final validation: once)."""

    digest = hashlib.sha256()
    byte_count = 0
    line_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
            line_count += chunk.count(b"\n")
    stat = path.stat()
    if byte_count != stat.st_size:
        raise core.StochasticBootstrapGenerationError(
            "generation WAL changed while hashing"
        )
    return {
        "digest": digest,
        "bytes": byte_count,
        "rows": line_count,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
    }


def _matching_cached_wal(path: Path) -> dict[str, Any] | None:
    stat = path.stat()
    resolved = path.expanduser().resolve()
    cached = _WAL_CACHE.get(resolved)
    if cached is not None:
        if cached["device"] != stat.st_dev or cached["inode"] != stat.st_ino:
            raise core.StochasticBootstrapGenerationError(
                "generation WAL file identity changed during the active process"
            )
        if cached["bytes"] == stat.st_size and cached["mtime_ns"] != stat.st_mtime_ns:
            raise core.StochasticBootstrapGenerationError(
                "generation WAL contents changed in place during the active process"
            )
        return cached
    # os.replace(progress, generations) preserves the inode; transfer the
    # rolling digest without rereading the sealed file.
    for old_path, value in list(_WAL_CACHE.items()):
        if value["device"] == stat.st_dev and value["inode"] == stat.st_ino:
            _WAL_CACHE.pop(old_path, None)
            _WAL_CACHE[resolved] = value
            return value
    return None


def _wal_snapshot(path: Path, results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return current SHA/bytes, extending the digest by at most one row."""

    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise core.StochasticBootstrapGenerationError(
            f"generation WAL is a symlink: {unresolved}"
        )
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise core.StochasticBootstrapGenerationError(
            f"generation WAL is missing or unsafe: {resolved}"
        )
    stat = resolved.stat()
    cached = _matching_cached_wal(resolved)
    if cached is None:
        if stat.st_size == 0:
            cached = {
                "digest": hashlib.sha256(),
                "bytes": 0,
                "rows": 0,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
            }
        else:
            cached = _scan_wal(resolved)
        _WAL_CACHE[resolved] = cached

    if cached["bytes"] != stat.st_size:
        can_extend_one = (
            cached["rows"] + 1 == len(results)
            and cached["bytes"] < stat.st_size
            and results
        )
        if can_extend_one:
            encoded = (core._canonical_json(results[-1]) + "\n").encode("utf-8")
            if cached["bytes"] + len(encoded) == stat.st_size:
                with resolved.open("rb") as handle:
                    handle.seek(cached["bytes"])
                    delta = handle.read()
                if delta != encoded:
                    raise core.StochasticBootstrapGenerationError(
                        "generation WAL delta differs from the canonical persisted row"
                    )
                cached["digest"].update(delta)
                cached["bytes"] += len(encoded)
                cached["rows"] += 1
                cached["mtime_ns"] = stat.st_mtime_ns
            else:
                raise core.StochasticBootstrapGenerationError(
                    "generation WAL size delta is not exactly one canonical row"
                )
        else:
            raise core.StochasticBootstrapGenerationError(
                "generation WAL was truncated or changed outside one-row append"
            )
    if cached["rows"] not in {len(results), len(results) + 1}:
        raise core.StochasticBootstrapGenerationError(
            "generation WAL row count differs from the in-memory canonical prefix"
        )
    return {
        "path": str(resolved),
        "sha256": cached["digest"].hexdigest(),
        "bytes": cached["bytes"],
    }


def _profile_sha256_file(path: Path) -> str:
    unresolved = path.expanduser()
    if _is_generation_wal(unresolved) and unresolved.is_symlink():
        raise core.StochasticBootstrapGenerationError(
            f"generation WAL is a symlink: {unresolved}"
        )
    resolved = unresolved.resolve()
    if _is_generation_wal(resolved) and resolved.is_file():
        cached = _matching_cached_wal(resolved)
        if cached is None:
            cached = _scan_wal(resolved)
            _WAL_CACHE[resolved] = cached
        if cached["bytes"] == resolved.stat().st_size:
            return str(cached["digest"].hexdigest())
    return _ORIGINAL_SHA256_FILE(resolved)


def _compact_completion_matrix(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    last = rows[-1] if rows else None
    completed_full_samples, partial_replicates = divmod(len(rows), len(REPLICATE_SEEDS))
    current_sample_replicates = (
        0
        if last is None
        else (partial_replicates if partial_replicates else len(REPLICATE_SEEDS))
    )
    return {
        "encoding": "canonical-prefix-count-and-last-tuple-v1",
        "completed_cases": len(rows),
        "completed_full_samples": completed_full_samples,
        "current_sample_replicates": current_sample_replicates,
        "last_tuple": None
        if last is None
        else {
            "model_id": last.get("model_id"),
            "sample_id": last.get("sample_id"),
            "replicate_id": last.get("replicate_id"),
        },
    }


def _profile_state_payload(
    *,
    status: str,
    model_id: str,
    results: Sequence[Mapping[str, Any]],
    expected_cases: int,
    resume_count: int,
    progress_path: Path,
    **extra: Any,
) -> dict[str, Any]:
    binding = _wal_snapshot(progress_path, results)
    return {
        "schema_version": core.STATE_SCHEMA_VERSION,
        "status": status,
        "updated_at_utc": core._utc_now(),
        "evaluation_id": core.EVALUATION_ID,
        "model_id": model_id,
        "completed_cases": len(results),
        "expected_cases": expected_cases,
        "resume_count": resume_count,
        "completion_matrix": _compact_completion_matrix(results),
        "partial_results": binding,
        "wal_strategy": {
            "schema_version": "incremental-generation-wal-sha256-v1",
            "append_flush_fsync_per_row": True,
            "rolling_sha256_and_bytes": True,
            "compact_prefix_state": True,
            "resume_full_scan_once": True,
        },
        **extra,
    }


def _profile_validate_resume_state(*args: Any, **kwargs: Any) -> int:
    state = kwargs.get("state")
    rows = kwargs.get("rows")
    if not isinstance(state, Mapping) or not isinstance(rows, Sequence):
        raise core.StochasticBootstrapGenerationError("invalid merged resume arguments")
    result = _ORIGINAL_VALIDATE_RESUME_STATE(*args, **kwargs)
    recorded = int(state["completed_cases"])
    if state.get("completion_matrix") != _compact_completion_matrix(rows[:recorded]):
        raise core.StochasticBootstrapGenerationError(
            "compact resume completion prefix drift"
        )
    expected_strategy = {
        "schema_version": "incremental-generation-wal-sha256-v1",
        "append_flush_fsync_per_row": True,
        "rolling_sha256_and_bytes": True,
        "compact_prefix_state": True,
        "resume_full_scan_once": True,
    }
    if state.get("wal_strategy") != expected_strategy:
        raise core.StochasticBootstrapGenerationError("resume WAL strategy drift")
    return result


def _reset_wal_cache() -> None:
    """Clear process-local rolling state (used between tests/process resumes)."""

    _WAL_CACHE.clear()


def _sample_metadata(sample: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in PROFILE_METADATA_KEYS if key not in sample]
    if missing:
        raise core.StochasticBootstrapGenerationError(
            f"merged sample source metadata is incomplete: {missing}"
        )
    return {key: copy.deepcopy(sample[key]) for key in PROFILE_METADATA_KEYS}


def source_metadata(
    row: Mapping[str, Any], source_binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the exact metadata copied to samples and generation rows."""

    metadata = contract.row_source_metadata(row)
    metadata.update(
        {
            "source_release": copy.deepcopy(dict(source_binding)),
            "research_scope": RESEARCH_SCOPE,
            "transport_split_role": TRANSPORT_SPLIT_ROLE,
            "not_all_held_out": True,
        }
    )
    return metadata


def _profile_sampling_contract() -> dict[str, Any]:
    if _ACTIVE_SOURCES is None:
        raise core.StochasticBootstrapGenerationError(
            "merged profile is not configured"
        )
    return {
        **_ORIGINAL_SAMPLING_CONTRACT(),
        "research_scope": RESEARCH_SCOPE,
        "transport_split_role": TRANSPORT_SPLIT_ROLE,
        "not_all_held_out": True,
        "legacy_core_evaluation_scope_label": (
            "formal_full_test_is_transport_only_for_this_profile"
        ),
        "source_role_counts": copy.deepcopy(dict(_ACTIVE_SOURCES.source_role_counts)),
        "cp318_selection_exposure": copy.deepcopy(
            dict(_ACTIVE_SOURCES.cp318_selection_exposure)
        ),
    }


def _profile_build_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
    sample = kwargs.get("sample")
    if not isinstance(sample, Mapping):
        raise core.StochasticBootstrapGenerationError("merged result has no sample")
    row = _ORIGINAL_BUILD_RESULT(*args, **kwargs)
    row.update(_sample_metadata(sample))
    return row


def _profile_validate_result(row: Mapping[str, Any], *args: Any, **kwargs: Any) -> None:
    sample = kwargs.get("sample")
    if not isinstance(sample, Mapping):
        raise core.StochasticBootstrapGenerationError("merged result has no sample")
    _ORIGINAL_VALIDATE_RESULT(row, *args, **kwargs)
    expected = _sample_metadata(sample)
    for key, value in expected.items():
        if row.get(key) != value:
            raise core.StochasticBootstrapGenerationError(
                f"merged stochastic row metadata drift at {key}"
            )


def _validate_sample_profile(manifest: Mapping[str, Any]) -> None:
    if _ACTIVE_SOURCES is None:
        raise core.StochasticBootstrapGenerationError(
            "merged profile is not configured"
        )
    profile = manifest.get("merged_core8_profile")
    if not isinstance(profile, Mapping):
        raise core.StochasticBootstrapGenerationError(
            "sample manifest has no merged Core8 profile"
        )
    expected_profile = profile_manifest_payload(_ACTIVE_SOURCES)
    if dict(profile) != expected_profile:
        raise core.StochasticBootstrapGenerationError(
            "sample manifest merged Core8 profile drift"
        )
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != contract.EXPECTED_ROWS:
        raise core.StochasticBootstrapGenerationError(
            "merged sample inventory is not N2048"
        )
    observed: dict[str, list[str]] = defaultdict(list)
    for expected_line, sample in enumerate(samples, 1):
        if not isinstance(sample, Mapping):
            raise core.StochasticBootstrapGenerationError(
                "merged sample is not an object"
            )
        sample_id = sample.get("sample_id")
        if (
            not isinstance(sample_id, str)
            or sample_id not in _EXPECTED_SAMPLE_METADATA
            or sample.get("line_number") != expected_line
        ):
            raise core.StochasticBootstrapGenerationError(
                f"merged sample identity/order drift at line {expected_line}"
            )
        expected = _EXPECTED_SAMPLE_METADATA[sample_id]
        for key, value in expected.items():
            if sample.get(key) != value:
                raise core.StochasticBootstrapGenerationError(
                    f"merged sample metadata drift at {key} (line {expected_line})"
                )
        if sample.get("meeting_id") != sample.get("meeting_end_date"):
            raise core.StochasticBootstrapGenerationError(
                "merged sample meeting cluster is not official end date"
            )
        observed[str(sample["meeting_id"])].append(str(sample["topic"]))
    if tuple(sorted(observed)) != _ACTIVE_SOURCES.meeting_ids:
        raise core.StochasticBootstrapGenerationError("merged meeting inventory drift")
    if any(tuple(topics) != contract.CORE_TOPICS for topics in observed.values()):
        raise core.StochasticBootstrapGenerationError(
            "merged per-meeting Core8 order drift"
        )


def profile_manifest_payload(
    sources: contract.HarmonizedSources,
) -> dict[str, Any]:
    """Return the exact profile block sealed in the sample manifest."""

    return {
        "schema_version": PROFILE_SCHEMA,
        "research_scope": RESEARCH_SCOPE,
        "transport_split_role": TRANSPORT_SPLIT_ROLE,
        "not_all_held_out": True,
        "legacy_core_evaluation_scope_label": (
            "formal_full_test_is_transport_only_for_this_profile"
        ),
        "source_releases": dict(sources.source_bindings),
        "evidence_cutoff_policy": contract.EVIDENCE_CUTOFF_POLICY,
        "meeting_cluster_date": "official_meeting_end_date",
        "ordering": "meeting_end_date_then_frozen_core8_topic_then_replicate_id",
        "topic_order": list(contract.CORE_TOPICS),
        "era_order": list(contract.ERA_ORDER),
        "meetings": contract.EXPECTED_MEETINGS,
        "prompts": contract.EXPECTED_ROWS,
        "replicates": len(REPLICATE_SEEDS),
        "models": list(core.MODEL_ORDER),
        "rows_per_model": contract.EXPECTED_ROWS * len(REPLICATE_SEEDS),
        "total_generation_rows": (
            contract.EXPECTED_ROWS * len(REPLICATE_SEEDS) * len(core.MODEL_ORDER)
        ),
        "topic_counts": dict(sources.topic_counts),
        "era_counts": dict(sources.era_counts),
        "source_role_counts": copy.deepcopy(dict(sources.source_role_counts)),
        "sensitivity_meetings": list(sources.sensitivity_meetings),
        "meeting_document_count": (
            contract.EXPECTED_MEETINGS * len(REPLICATE_SEEDS) * len(core.MODEL_ORDER)
        ),
        "cp318_selection_exposure": copy.deepcopy(
            dict(sources.cp318_selection_exposure)
        ),
    }


def _profile_load_sample_manifest(
    path: Path, expected_sha256: str
) -> tuple[Mapping[str, Any], str]:
    manifest, observed_sha = _ORIGINAL_LOAD_SAMPLE_MANIFEST(path, expected_sha256)
    _validate_sample_profile(manifest)
    return manifest, observed_sha


def _profile_source_hashes(
    manifest: Mapping[str, Any], manifest_sha256: str
) -> dict[str, Any]:
    if _ACTIVE_SOURCES is None:
        raise core.StochasticBootstrapGenerationError(
            "merged profile is not configured"
        )
    observed_profile = manifest.get("merged_core8_profile")
    if not isinstance(observed_profile, Mapping) or observed_profile.get(
        "source_releases"
    ) != dict(_ACTIVE_SOURCES.source_bindings):
        raise core.StochasticBootstrapGenerationError(
            "sample/source release binding drift"
        )
    hashes = _ORIGINAL_SOURCE_HASHES(manifest, manifest_sha256)
    hashes["harmonized_source_releases"] = copy.deepcopy(
        dict(_ACTIVE_SOURCES.source_bindings)
    )
    hashes["implementation_sources"]["merged_profile_runner"] = core._file_binding(
        Path(__file__).resolve()
    )
    hashes["implementation_sources"]["merged_data_contract"] = core._file_binding(
        Path(str(contract.__file__)).resolve()
    )
    return hashes


def configure_profile(
    *,
    pre_release_manifest: Path = contract.PRE_RELEASE_MANIFEST,
    post_release_manifest: Path = contract.POST_RELEASE_MANIFEST,
    pre_release_sha256: str | None = None,
    post_release_sha256: str | None = None,
) -> contract.HarmonizedSources:
    """Apply the frozen N2048/K10 profile and deep-bind both source releases."""

    global _ACTIVE_SOURCES, _EXPECTED_SAMPLE_METADATA
    sources = contract.load_harmonized_sources(
        pre_release_manifest=pre_release_manifest,
        post_release_manifest=post_release_manifest,
        pre_release_sha256=pre_release_sha256,
        post_release_sha256=post_release_sha256,
    )
    if _ACTIVE_SOURCES is not None and _canonical(
        _ACTIVE_SOURCES.source_bindings
    ) != _canonical(sources.source_bindings):
        raise core.StochasticBootstrapGenerationError(
            "refusing to reconfigure merged profile to different source releases"
        )
    expected_metadata: dict[str, dict[str, Any]] = {}
    for row in sources.rows:
        binding = sources.source_bindings[str(row["era"])]
        sample_id = contract.merged_sample_id(row, binding)
        metadata = source_metadata(row, binding)
        if sample_id in expected_metadata:
            raise core.StochasticBootstrapGenerationError(
                f"derived merged sample ID collision: {sample_id}"
            )
        expected_metadata[sample_id] = metadata

    core.EVALUATION_ID = "chk3-beta-core8-merged-1993-2025-n2048-k10-v1"
    core.SAMPLE_MANIFEST_SCHEMA_VERSION = (
        "chk3-beta-core8-merged-stochastic-n2048-k10-samples-v1"
    )
    core.ROW_SCHEMA_VERSION = "chk3-beta-core8-merged-generation-row-v1"
    core.RUN_MANIFEST_SCHEMA_VERSION = (
        "chk3-beta-core8-merged-generation-run-manifest-v1"
    )
    core.SUITE_MANIFEST_SCHEMA_VERSION = "chk3-beta-core8-merged-generation-suite-v1"
    core.STATE_SCHEMA_VERSION = "chk3-beta-core8-merged-generation-state-v1"
    core.REPLICATE_SEEDS = REPLICATE_SEEDS
    core.EXPECTED_TEST_ROWS = contract.EXPECTED_ROWS
    core.EXPECTED_MEETING_IDS = sources.meeting_ids
    core.EXPECTED_IDENTITY_ROWS = 0
    core.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    core.MEETING_ID_RE = re.compile(
        r"^chk3-beta-core8-((?:19|20)\d{2}-\d{2}-\d{2})-[0-9a-f]{24}$"
    )
    core.GPU_LOCK_PATH = Path(
        "/tmp/fomc_trainer_chk3_beta_core8_merged_n2048_k10_gpu0.lock"
    )
    _ACTIVE_SOURCES = sources
    _EXPECTED_SAMPLE_METADATA = expected_metadata
    core.build_stochastic_result = _profile_build_result
    core.validate_stochastic_result = _profile_validate_result
    core.load_full_test_sample_manifest = _profile_load_sample_manifest
    core._source_hashes = _profile_source_hashes
    core._sha256_file = _profile_sha256_file
    core._completion_matrix = _compact_completion_matrix
    core._state_payload = _profile_state_payload
    core._validate_resume_state = _profile_validate_resume_state
    core._sampling_contract = _profile_sampling_contract
    return sources


def _profile_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--pre-release", type=Path, default=contract.PRE_RELEASE_MANIFEST
    )
    parser.add_argument(
        "--post-release", type=Path, default=contract.POST_RELEASE_MANIFEST
    )
    parser.add_argument("--pre-release-sha256")
    parser.add_argument("--post-release-sha256", required=True)
    return parser.parse_known_args(argv)


def _reject_gpu_lock_override(argv: Sequence[str]) -> None:
    if any(
        token == "--gpu-lock-path" or token.startswith("--gpu-lock-path=")
        for token in argv
    ):
        raise core.StochasticBootstrapGenerationError(
            "merged profile uses a fixed GPU0-exclusive lock; "
            "--gpu-lock-path cannot be overridden"
        )


def main(argv: Sequence[str] | None = None) -> int:
    profile_args, remaining = _profile_args(argv)
    try:
        _reject_gpu_lock_override(remaining)
        configure_profile(
            pre_release_manifest=profile_args.pre_release,
            post_release_manifest=profile_args.post_release,
            pre_release_sha256=profile_args.pre_release_sha256,
            post_release_sha256=profile_args.post_release_sha256,
        )
    except (
        contract.MergedCore8ContractError,
        core.StochasticBootstrapGenerationError,
        OSError,
        ValueError,
    ) as exc:
        print(
            _canonical(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 1
    prior_argv = sys.argv
    try:
        sys.argv = [prior_argv[0], *remaining]
        return core.main()
    finally:
        sys.argv = prior_argv


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_SEED",
    "PROFILE_METADATA_KEYS",
    "PROFILE_SCHEMA",
    "RESEARCH_SCOPE",
    "REPLICATE_SEEDS",
    "_compact_completion_matrix",
    "_profile_sha256_file",
    "_profile_state_payload",
    "_reset_wal_cache",
    "configure_profile",
    "main",
    "profile_manifest_payload",
    "source_metadata",
]
