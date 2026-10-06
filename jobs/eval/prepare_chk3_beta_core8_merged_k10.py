"""Prepare the sealed N2048 x K10 merged Core8 generation cohort."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jobs.eval import chk3_beta_core8_merged_contract as contract
from jobs.eval import eval_chk3_beta_core8_merged_stochastic_k10 as profile
from jobs.eval import eval_chk3_stochastic_bootstrap_generation as core
from jobs.eval.prepare_chk3_external_holdout_smoke import (
    DEFAULT_CONFIG,
    DEFAULT_TOKENIZER,
    ExternalSmokeError,
    _binding,
    _write_new_json,
    _write_new_jsonl,
)
from jobs.retrain_v2 import probe_chk3_sft_degeneration as native_probe
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import seal_manifest


SCHEMA_VERSION = "chk3-beta-core8-merged-k10-preparation-v1"
COMPATIBILITY_RELEASE_ID = contract.MERGED_RELEASE_ID + "-compatibility"


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _build_compatibility_rows(
    sources: contract.HarmonizedSources, tokenizer: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    data_rows: list[dict[str, Any]] = []
    row_manifest: list[dict[str, Any]] = []
    sample_metadata: dict[str, dict[str, Any]] = {}
    for line_number, source in enumerate(sources.rows, 1):
        source_binding = sources.source_bindings[str(source["era"])]
        sample_id = contract.merged_sample_id(source, source_binding)
        prompt = str(source["prompt"])
        analysis = native_probe.extract_source_analysis(prompt)
        reference = str(source["reference_minutes"])
        if analysis != source["source_analysis"]:
            raise ExternalSmokeError(
                f"source analysis drift at merged line {line_number}"
            )
        response = "Reference transport wrapper.\n</think>\n" + reference
        metadata = profile.source_metadata(source, source_binding)
        data_rows.append(
            {
                "prompt": prompt,
                "response": response,
                "sample_id": sample_id,
                **copy.deepcopy(metadata),
            }
        )
        row_manifest.append(
            {
                "sample_id": sample_id,
                "split": "test",
                # The durable runner requires a test-only compatibility release.
                # The real historical role is preserved separately and per row.
                "source_split": "test",
                "origin_source_split": metadata["source_split"],
                "original_qa_split": metadata["original_qa_split"],
                "prompt_sha256": native_probe.common_probe.sha256_text(prompt),
                "response_sha256": native_probe.common_probe.sha256_text(response),
                "analysis_sha256": native_probe.common_probe.sha256_text(analysis),
                "completion_tokens": len(
                    tokenizer.encode(reference, add_special_tokens=False)
                ),
                "era": metadata["era"],
                "meeting_end_date": metadata["meeting_end_date"],
                "topic": metadata["topic"],
                "topic_order": metadata["topic_order"],
                "source_sample_id": metadata["source_sample_id"],
                "source_core8_line_number": metadata["source_core8_line_number"],
                "source_release_manifest_sha256": source_binding["sha256"],
            }
        )
        if sample_id in sample_metadata:
            raise ExternalSmokeError(f"merged sample ID collision: {sample_id}")
        sample_metadata[sample_id] = metadata
    if len(data_rows) != contract.EXPECTED_ROWS:
        raise ExternalSmokeError("compatibility row closure is not N2048")
    return data_rows, row_manifest, sample_metadata


def prepare(
    *,
    pre_release_manifest: Path,
    post_release_manifest: Path,
    pre_release_sha256: str | None,
    post_release_sha256: str | None,
    tokenizer_path: Path,
    training_config: Path,
    output_dir: Path,
) -> dict[str, Any]:
    unresolved_output = output_dir.expanduser()
    if unresolved_output.is_symlink():
        raise ExternalSmokeError(
            f"refusing symlink output directory: {unresolved_output}"
        )
    output_dir = unresolved_output.resolve()
    if output_dir.exists():
        raise ExternalSmokeError(f"refusing to reuse output directory: {output_dir}")
    sources = profile.configure_profile(
        pre_release_manifest=pre_release_manifest,
        post_release_manifest=post_release_manifest,
        pre_release_sha256=pre_release_sha256,
        post_release_sha256=post_release_sha256,
    )
    tokenizer_path = tokenizer_path.expanduser().resolve()
    training_config = training_config.expanduser().resolve()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=True
    )
    data_rows, row_manifest, sample_metadata = _build_compatibility_rows(
        sources, tokenizer
    )
    data_path = output_dir / "compatibility_release/minutes_alignment/test.jsonl"
    row_path = (
        output_dir / "compatibility_release/minutes_alignment/manifests/test.jsonl"
    )
    release_path = output_dir / "compatibility_release/release_manifest.json"
    _write_new_jsonl(data_path, data_rows)
    _write_new_jsonl(row_path, row_manifest)
    compatibility_release = seal_manifest(
        {
            "schema_version": "chk3-minutes-training-release-v1",
            "release_id": COMPATIBILITY_RELEASE_ID,
            "immutable": True,
            "quality_status": "passed",
            "evaluation_only": True,
            "trainable": False,
            "checkpoint_selection_allowed": False,
            "promotable": False,
            "task_contract": contract.TASK_CONTRACT,
            "grain": "one_meeting_core8_topic_per_row",
            "split_counts": {"test": contract.EXPECTED_ROWS},
            "evidence_cutoff_policy": contract.EVIDENCE_CUTOFF_POLICY,
            "meeting_cluster_date": "official_meeting_end_date",
            "research_scope": profile.RESEARCH_SCOPE,
            "transport_split_role": profile.TRANSPORT_SPLIT_ROLE,
            "not_all_held_out": True,
            "legacy_core_test_split_is_transport_only": True,
            "source_releases": copy.deepcopy(dict(sources.source_bindings)),
            "source_role_counts": copy.deepcopy(dict(sources.source_role_counts)),
            "cp318_selection_exposure": copy.deepcopy(
                dict(sources.cp318_selection_exposure)
            ),
            "ordering": {
                "rows": "meeting_end_date_then_frozen_core8_topic",
                "topic_order": list(contract.CORE_TOPICS),
            },
            "files": {
                "minutes_alignment/test.jsonl": {
                    **_binding(data_path, rows=contract.EXPECTED_ROWS),
                    "path": "minutes_alignment/test.jsonl",
                },
                "minutes_alignment/manifests/test.jsonl": {
                    **_binding(row_path, rows=contract.EXPECTED_ROWS),
                    "path": "minutes_alignment/manifests/test.jsonl",
                },
            },
        }
    )
    _write_new_json(release_path, compatibility_release)

    sample_manifest = core.build_full_test_sample_manifest(
        test_data=data_path,
        test_row_manifest=row_path,
        release_manifest=release_path,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        training_config=training_config,
    )
    sample_manifest = dict(sample_manifest)
    sample_manifest.pop("integrity", None)
    for sample in sample_manifest["samples"]:
        sample_id = str(sample["sample_id"])
        metadata = sample_metadata.get(sample_id)
        if metadata is None:
            raise ExternalSmokeError(f"sample metadata missing for {sample_id}")
        sample.update(copy.deepcopy(metadata))
    sample_manifest["merged_core8_profile"] = profile.profile_manifest_payload(sources)
    sample_manifest["preparation_lineage"] = {
        "preparer": _binding(Path(__file__).resolve()),
        "profile_runner": _binding(Path(str(profile.__file__)).resolve()),
        "data_contract": _binding(Path(str(contract.__file__)).resolve()),
        "compatibility_release": {
            **_binding(release_path),
            "payload_sha256": compatibility_release["integrity"]["payload_sha256"],
        },
        "source_model_output_used": False,
        "missing_topic_imputation": False,
        "placeholder_rows": 0,
    }
    sample_manifest = seal_manifest(sample_manifest)
    sample_path = output_dir / "samples_n2048_k10.json"
    _write_new_json(sample_path, sample_manifest)
    loaded, observed_sample_sha = core.load_full_test_sample_manifest(
        sample_path, sha256_file(sample_path)
    )
    coverage = {
        "meetings": contract.EXPECTED_MEETINGS,
        "prompts": contract.EXPECTED_ROWS,
        "topics_per_meeting": len(contract.CORE_TOPICS),
        "replicates": len(profile.REPLICATE_SEEDS),
        "models": len(core.MODEL_ORDER),
        "rows_per_model": contract.EXPECTED_ROWS * len(profile.REPLICATE_SEEDS),
        "total_generation_rows": (
            contract.EXPECTED_ROWS
            * len(profile.REPLICATE_SEEDS)
            * len(core.MODEL_ORDER)
        ),
        "meeting_documents": (
            contract.EXPECTED_MEETINGS
            * len(profile.REPLICATE_SEEDS)
            * len(core.MODEL_ORDER)
        ),
        "topic_counts": dict(sources.topic_counts),
        "era_counts": dict(sources.era_counts),
        "original_post_split_counts": dict(
            sorted(
                Counter(
                    str(row["source_split"])
                    for row in sources.rows
                    if row["era"] == "post2008_chk3_release"
                ).items()
            )
        ),
        "original_post_qa_split_counts": dict(
            sorted(
                Counter(
                    str(row["original_qa_split"])
                    for row in sources.rows
                    if row["era"] == "post2008_chk3_release"
                ).items()
            )
        ),
        "sensitivity_meetings": len(sources.sensitivity_meetings),
        "cp318_selection_exposed_meetings": 9,
        "cp318_selection_exposed_prompts": 72,
        "cp318_selection_exposed_generation_rows_per_model": 720,
        "normalized_identity_rows": sum(
            bool(row["normalized_identity"]) for row in sample_manifest["samples"]
        ),
    }
    preparation = seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "sample_manifest": {
                **_binding(sample_path),
                "payload_sha256": loaded["integrity"]["payload_sha256"],
            },
            "compatibility_release": {
                **_binding(release_path),
                "payload_sha256": compatibility_release["integrity"]["payload_sha256"],
            },
            "source_releases": copy.deepcopy(dict(sources.source_bindings)),
            "coverage": coverage,
            "validation": {
                "deep_profile_loader": "passed",
                "observed_sample_sha256": observed_sample_sha,
                "official_start_date_d1_only": True,
                "date_overlap": 0,
                "duplicate_meeting_topic_keys": 0,
                "missing_topic_rows": 0,
                "placeholder_rows": 0,
                "model_output_used": False,
            },
        }
    )
    preparation_path = output_dir / "preparation.json"
    _write_new_json(preparation_path, preparation)
    return preparation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pre-release", type=Path, default=contract.PRE_RELEASE_MANIFEST
    )
    parser.add_argument(
        "--post-release", type=Path, default=contract.POST_RELEASE_MANIFEST
    )
    parser.add_argument("--pre-release-sha256")
    parser.add_argument("--post-release-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, default=core.ROOT / DEFAULT_TOKENIZER)
    parser.add_argument(
        "--training-config", type=Path, default=core.ROOT / DEFAULT_CONFIG
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare(
            pre_release_manifest=args.pre_release,
            post_release_manifest=args.post_release,
            pre_release_sha256=args.pre_release_sha256,
            post_release_sha256=args.post_release_sha256,
            tokenizer_path=args.tokenizer,
            training_config=args.training_config,
            output_dir=args.output_dir,
        )
    except (
        contract.MergedCore8ContractError,
        ExternalSmokeError,
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
            )
        )
        return 1
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPATIBILITY_RELEASE_ID",
    "SCHEMA_VERSION",
    "_build_compatibility_rows",
    "prepare",
]
