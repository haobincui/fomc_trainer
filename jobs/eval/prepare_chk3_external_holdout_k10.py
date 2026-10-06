"""Prepare the frozen N128 x K10 external CHK3 stochastic evaluation inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jobs.eval import eval_chk3_external_holdout_stochastic_k10 as profile
from jobs.eval import eval_chk3_stochastic_bootstrap_generation as core
from jobs.eval.prepare_chk3_external_holdout_smoke import (
    DEFAULT_CONFIG,
    DEFAULT_TOKENIZER,
    ExternalSmokeError,
    _binding,
    _read_json,
    _read_jsonl,
    _write_new_json,
    _write_new_jsonl,
)
from jobs.retrain_v2 import probe_chk3_sft_degeneration as native_probe
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import seal_manifest, validate_manifest_integrity


CORE8 = profile.RELEASE_MANIFEST.parent / "panels/core8.jsonl"
SCHEMA_VERSION = "chk3-external-holdout-k10-preparation-v1"


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def _select_balanced(rows: list[dict[str, Any]], release_sha: str) -> list[dict[str, Any]]:
    by_meeting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_meeting[str(row["meeting_start_date"])].append(row)
    if len(by_meeting) != 128 or any(len(items) != 8 for items in by_meeting.values()):
        raise ExternalSmokeError("core8 is not exactly 128 meetings x 8 topics")
    all_topics = sorted({str(row["topic"]) for row in rows})
    if len(all_topics) != 8:
        raise ExternalSmokeError("core8 topic inventory is not N8")
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for meeting in sorted(by_meeting):
        minimum = min(counts[topic] for topic in all_topics)
        eligible = [row for row in by_meeting[meeting] if counts[str(row["topic"])] == minimum]
        chosen = min(
            eligible,
            key=lambda row: hashlib.sha256(
                f"external-n128-balanced-topic-v1|{release_sha}|{meeting}|{row['topic']}".encode()
            ).hexdigest(),
        )
        counts[str(chosen["topic"])] += 1
        selected.append(chosen)
    if set(counts.values()) != {16}:
        raise ExternalSmokeError(f"topic allocation is not exactly balanced: {counts}")
    return selected


def prepare(*, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise ExternalSmokeError(f"refusing to reuse output directory: {output_dir}")
    meetings = profile.configure_profile()
    source_release = _read_json(profile.RELEASE_MANIFEST)
    source_payload = validate_manifest_integrity(source_release)
    if (
        source_release.get("schema_version")
        != "chk3-external-evaluation-release-v1"
        or source_release.get("status") != "passed"
        or source_release.get("evaluation_only") is not True
        or source_release.get("trainable") is not False
        or source_release.get("checkpoint_selection_allowed") is not False
        or source_release.get("promotable") is not False
    ):
        raise ExternalSmokeError("source external-release scope/status drift")
    core8_record = source_release.get("files", {}).get("panels/core8.jsonl")
    if not isinstance(core8_record, dict):
        raise ExternalSmokeError("source release does not bind core8")
    source_rows = _read_jsonl(CORE8)
    if (
        core8_record.get("sha256") != sha256_file(CORE8)
        or core8_record.get("rows") != 1024
        or len(source_rows) != 1024
    ):
        raise ExternalSmokeError("core8 source binding changed")
    selected = _select_balanced(source_rows, profile.RELEASE_MANIFEST_SHA256)
    if tuple(str(row["meeting_start_date"]) for row in selected) != meetings:
        raise ExternalSmokeError("selected meeting order differs from frozen roster")

    from transformers import AutoTokenizer

    tokenizer_path = (core.ROOT / DEFAULT_TOKENIZER).resolve()
    training_config = (core.ROOT / DEFAULT_CONFIG).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=True
    )
    compatibility_rows: list[dict[str, Any]] = []
    row_manifest: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, str]] = {}
    for index, source in enumerate(selected, 1):
        prompt = str(source["prompt"])
        analysis = native_probe.extract_source_analysis(prompt)
        reference = str(source["reference_minutes"])
        if (
            native_probe.common_probe.sha256_text(prompt) != source["prompt_sha256"]
            or native_probe.common_probe.sha256_text(analysis)
            != source["source_analysis_sha256"]
            or native_probe.common_probe.sha256_text(reference)
            != source["reference_minutes_sha256"]
        ):
            raise ExternalSmokeError(f"source row hash drift at selected row {index}")
        meeting = str(source["meeting_start_date"])
        sample_suffix = hashlib.sha256(str(source["sample_id"]).encode()).hexdigest()[:16]
        sample_id = f"chk3-external-{meeting}-{sample_suffix}"
        response = "Reference transport wrapper.\n</think>\n" + reference
        compatibility_rows.append(
            {
                "prompt": prompt,
                "response": response,
                "external_source_sample_id": source["sample_id"],
                "meeting_start_date": meeting,
                "topic": source["topic"],
            }
        )
        row_manifest.append(
            {
                "sample_id": sample_id,
                "split": "test",
                "source_split": "test",
                "prompt_sha256": native_probe.common_probe.sha256_text(prompt),
                "response_sha256": native_probe.common_probe.sha256_text(response),
                "analysis_sha256": native_probe.common_probe.sha256_text(analysis),
                "completion_tokens": len(
                    tokenizer.encode(reference, add_special_tokens=False)
                ),
                "source_core8_sample_id": source["sample_id"],
                "source_core8_line_number": source_rows.index(source) + 1,
            }
        )
        metadata[sample_id] = {
            "meeting_start_date": meeting,
            "topic": str(source["topic"]),
            "source_core8_sample_id": str(source["sample_id"]),
        }

    data_path = output_dir / "compatibility_release/minutes_alignment/test.jsonl"
    row_path = output_dir / "compatibility_release/minutes_alignment/manifests/test.jsonl"
    release_path = output_dir / "compatibility_release/release_manifest.json"
    _write_new_jsonl(data_path, compatibility_rows)
    _write_new_jsonl(row_path, row_manifest)
    release = seal_manifest(
        {
            "schema_version": "chk3-minutes-training-release-v1",
            "release_id": "chk3-external-holdout-1993-2008-balanced-n128-v1",
            "immutable": True,
            "quality_status": "passed",
            "evaluation_only": True,
            "trainable": False,
            "checkpoint_selection_allowed": False,
            "promotable": False,
            "split_counts": {"test": 128},
            "files": {
                "minutes_alignment/test.jsonl": {
                    **_binding(data_path, rows=128),
                    "path": "minutes_alignment/test.jsonl",
                },
                "minutes_alignment/manifests/test.jsonl": {
                    **_binding(row_path, rows=128),
                    "path": "minutes_alignment/manifests/test.jsonl",
                },
            },
            "source_release": {
                "path": str(profile.RELEASE_MANIFEST),
                "sha256": profile.RELEASE_MANIFEST_SHA256,
                "payload_sha256": source_payload,
            },
            "source_core8": _binding(CORE8, rows=1024),
            "selection": {
                "algorithm": "chronological-meeting-balanced-topic-hash-v1",
                "meetings": 128,
                "topics": 8,
                "rows_per_topic": 16,
                "model_output_independent": True,
            },
        }
    )
    _write_new_json(release_path, release)

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
        sample.update(metadata[str(sample["sample_id"])])
    sample_manifest["external_profile"] = {
        "source_release": {
            "path": str(profile.RELEASE_MANIFEST),
            "sha256": profile.RELEASE_MANIFEST_SHA256,
            "payload_sha256": source_payload,
        },
        "source_core8": _binding(CORE8, rows=1024),
        "profile_runner": _binding(Path(profile.__file__).resolve()),
        "preparer": _binding(Path(__file__).resolve()),
        "meeting_coverage": "all 128 regular FOMC meetings from 1993 through 2008",
        "topic_balance": dict(sorted(Counter(row["topic"] for row in selected).items())),
        "reference_role": "deterministic source-grounded Minutes-style target",
        "official_minutes_model_input": False,
    }
    sample_manifest = seal_manifest(sample_manifest)
    sample_path = output_dir / "samples_n128_k10.json"
    _write_new_json(sample_path, sample_manifest)
    loaded, observed_sha = core.load_full_test_sample_manifest(
        sample_path, sha256_file(sample_path)
    )
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
                "payload_sha256": release["integrity"]["payload_sha256"],
            },
            "coverage": {
                "meetings": 128,
                "prompts": 128,
                "replicates": 10,
                "models": 3,
                "rows_per_model": 1280,
                "total_generation_rows": 3840,
                "topic_counts": dict(
                    sorted(Counter(str(row["topic"]) for row in selected).items())
                ),
                "normalized_identity_rows": sum(
                    bool(row["normalized_identity"])
                    for row in sample_manifest["samples"]
                ),
            },
            "validation": {
                "deep_profile_loader": "passed",
                "observed_sample_sha256": observed_sha,
                "model_output_used_for_selection": False,
            },
        }
    )
    preparation_path = output_dir / "preparation.json"
    _write_new_json(preparation_path, preparation)
    return preparation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = prepare(output_dir=args.output_dir)
    except ExternalSmokeError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
