"""Prepare a sealed N=12 CHK3-native smoke from the 1993--2008 holdout.

The native three-model evaluator consumes the historical CHK3 ``prompt`` /
``response`` transport shape.  The external holdout instead stores the
source-grounded final Minutes reference explicitly.  This command creates a
versioned, evaluation-only compatibility view without modifying either source
release: ``response`` is a transport wrapper around the frozen final reference,
and no synthetic reasoning is used as model input or as a scored target.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import eval_chk3_native_three_model as native_eval
from jobs.retrain_v2 import probe_chk3_sft_degeneration as native_probe
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


DEFAULT_RELEASE = Path(
    "dataset/processed/retrain_v2/"
    "chk3_minutes_external_holdout_1993_2008_all_regular_v1/release_manifest.json"
)
DEFAULT_PANEL = DEFAULT_RELEASE.parent / "panels/expanded_all_available.jsonl"
DEFAULT_TOKENIZER = Path(
    "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
)
DEFAULT_CONFIG = Path(
    "configs/retrain_v2/"
    "chk3_minutes_sft_direct_chk1_cp200_full3ep_lr1e6_20260810.yaml"
)
OUTPUT_SCHEMA = "chk3-external-holdout-smoke-preparation-v1"


class ExternalSmokeError(RuntimeError):
    """The compatibility view cannot be built without breaking provenance."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalSmokeError(f"invalid JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExternalSmokeError(f"JSON artifact is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExternalSmokeError(f"cannot read JSONL: {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExternalSmokeError(f"invalid JSONL: {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ExternalSmokeError(f"JSONL row is not an object: {path}:{line_number}")
        result.append(row)
    return result


def _write_new_bytes(path: Path, body: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ExternalSmokeError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def _write_new_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    body = "".join(_canonical_json(dict(row)) + "\n" for row in rows).encode()
    _write_new_bytes(path, body)


def _normalised(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _punctuation_insensitive(text: str) -> str:
    return " ".join(
        "".join(
            character
            for character in unicodedata.normalize("NFKC", text).casefold()
            if not unicodedata.category(character).startswith("P")
        ).split()
    )


def _binding(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def prepare(
    *,
    release_manifest: Path,
    panel: Path,
    tokenizer_path: Path,
    training_config: Path,
    output_dir: Path,
) -> dict[str, Any]:
    release_manifest = release_manifest.expanduser().resolve()
    panel = panel.expanduser().resolve()
    tokenizer_path = tokenizer_path.expanduser().resolve()
    training_config = training_config.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ExternalSmokeError(f"output directory already exists: {output_dir}")

    release = _read_json(release_manifest)
    try:
        release_payload_sha = validate_manifest_integrity(release)
    except Exception as exc:
        raise ExternalSmokeError(f"release manifest integrity failed: {exc}") from exc
    if (
        release.get("schema_version") != "chk3-external-evaluation-release-v1"
        or release.get("status") != "passed"
        or release.get("evaluation_only") is not True
        or release.get("trainable") is not False
        or release.get("checkpoint_selection_allowed") is not False
        or release.get("promotable") is not False
    ):
        raise ExternalSmokeError("external release scope/schema is invalid")
    panel_record = (release.get("files") or {}).get(
        "panels/expanded_all_available.jsonl"
    )
    if not isinstance(panel_record, Mapping):
        raise ExternalSmokeError("release does not bind the expanded panel")
    rows = _read_jsonl(panel)
    if (
        panel.resolve()
        != (release_manifest.parent / str(panel_record.get("path"))).resolve()
        or panel_record.get("sha256") != sha256_file(panel)
        or panel_record.get("rows") != len(rows)
    ):
        raise ExternalSmokeError("expanded panel binding changed")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=True
    )
    try:
        system_prompt, suffix, config_sha = native_probe._load_prompt_contract(
            training_config
        )
    except native_probe.Chk3ProbeError as exc:
        raise ExternalSmokeError(str(exc)) from exc

    compatibility_rows: list[dict[str, Any]] = []
    row_manifest: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, str]] = set()
    for line_number, source in enumerate(rows, 1):
        required = (
            "sample_id",
            "meeting_id",
            "meeting_start_date",
            "topic",
            "prompt",
            "source_analysis",
            "reference_minutes",
            "prompt_sha256",
            "source_analysis_sha256",
            "reference_minutes_sha256",
        )
        if any(not isinstance(source.get(key), str) or not source[key] for key in required):
            raise ExternalSmokeError(f"external row is incomplete: line {line_number}")
        sample_id = str(source["sample_id"])
        key = (str(source["meeting_id"]), str(source["topic"]))
        if sample_id in seen_ids or key in seen_keys:
            raise ExternalSmokeError(f"duplicate sample/key: line {line_number}")
        seen_ids.add(sample_id)
        seen_keys.add(key)
        prompt = str(source["prompt"])
        analysis = native_probe.extract_source_analysis(prompt)
        reference = str(source["reference_minutes"])
        if (
            native_probe.common_probe.sha256_text(prompt) != source["prompt_sha256"]
            or native_probe.common_probe.sha256_text(analysis)
            != source["source_analysis_sha256"]
            or native_probe.common_probe.sha256_text(reference)
            != source["reference_minutes_sha256"]
            or analysis != source["source_analysis"]
        ):
            raise ExternalSmokeError(f"external row hash/text drift: line {line_number}")
        # This wrapper exists only to satisfy the historical evaluator transport;
        # the text before the boundary is never a scored reference.
        response = "Reference transport wrapper.\n</think>\n" + reference
        prompt_sha = native_probe.common_probe.sha256_text(prompt)
        response_sha = native_probe.common_probe.sha256_text(response)
        reference_tokens = len(tokenizer.encode(reference, add_special_tokens=False))
        try:
            prompt_ids = native_probe._prompt_ids(
                tokenizer,
                native_probe._messages(
                    {"prompt": prompt},
                    system_prompt=system_prompt,
                    user_prompt_suffix=suffix,
                ),
            )
        except native_probe.Chk3ProbeError as exc:
            raise ExternalSmokeError(str(exc)) from exc
        compatibility_rows.append(
            {
                "prompt": prompt,
                "response": response,
                "external_sample_id": sample_id,
                "meeting_start_date": source["meeting_start_date"],
                "topic": source["topic"],
            }
        )
        row_manifest.append(
            {
                "sample_id": sample_id,
                "split": "test",
                "source_split": "test",
                "prompt_sha256": prompt_sha,
                "response_sha256": response_sha,
                "analysis_sha256": native_probe.common_probe.sha256_text(analysis),
                "completion_tokens": reference_tokens,
                "external_panel_line_number": line_number,
            }
        )
        candidates.append(
            {
                "sample_id": sample_id,
                "line_number": line_number,
                "release_row_manifest_line_number": line_number,
                "prompt_sha256": prompt_sha,
                "response_sha256": response_sha,
                "analysis_sha256": native_probe.common_probe.sha256_text(analysis),
                "release_analysis_sha256": source["source_analysis_sha256"],
                "reference_minutes_sha256": source["reference_minutes_sha256"],
                "completion_tokens": reference_tokens,
                "analysis_reference_exact_identity": analysis == reference,
                "normalized_identity": _normalised(analysis) == _normalised(reference),
                "punctuation_insensitive_identity": (
                    _punctuation_insensitive(analysis)
                    == _punctuation_insensitive(reference)
                ),
                "source_numeric_multiset_sha256": native_probe._numeric_hash(analysis),
                "source_date_set_sha256": native_probe._date_hash(analysis),
                "source_numeric_occurrences": sum(
                    native_probe._numeric_values(analysis).values()
                ),
                "source_date_values": len(native_probe._date_values(analysis)),
                "prompt_token_count": len(prompt_ids),
                "meeting_start_date": source["meeting_start_date"],
                "topic": source["topic"],
                "external_panel_line_number": line_number,
            }
        )
    if len(candidates) != release.get("expanded_rows"):
        raise ExternalSmokeError("expanded row count changed")

    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = output_dir / "compatibility_dataset.jsonl"
    row_manifest_path = output_dir / "compatibility_row_manifest.jsonl"
    _write_new_jsonl(dataset_path, compatibility_rows)
    _write_new_jsonl(row_manifest_path, row_manifest)

    ordered = sorted(
        candidates,
        key=lambda row: (int(row["completion_tokens"]), str(row["sample_id"])),
    )
    boundaries = (0, len(ordered) // 3, (2 * len(ordered)) // 3, len(ordered))
    release_sha = sha256_file(release_manifest)
    selected: list[dict[str, Any]] = []
    for index, bucket in enumerate(("short", "medium", "long")):
        population = ordered[boundaries[index] : boundaries[index + 1]]
        ranked = sorted(
            population,
            key=lambda row: (
                native_probe.common_probe.sha256_text(
                    f"{native_eval.TASK_CONTRACT_ID}|{release_sha}|{row['sample_id']}"
                ),
                str(row["sample_id"]),
            ),
        )
        selected.extend(
            {
                **row,
                "length_bucket": bucket,
                "within_bucket_hash_rank": rank,
                "bucket_population_rows": len(population),
            }
            for rank, row in enumerate(ranked[:4])
        )

    sample_payload = {
        "schema_version": native_eval.SAMPLE_MANIFEST_SCHEMA_VERSION,
        "task_contract_id": native_eval.TASK_CONTRACT_ID,
        "release": {
            "path": str(release_manifest),
            "sha256": release_sha,
            "payload_sha256": release_payload_sha,
            "release_id": release.get("release_id"),
            "schema_version": release.get("schema_version"),
        },
        "dataset": {**_binding(dataset_path, rows=len(compatibility_rows)), "split": "test"},
        "release_row_manifest": _binding(row_manifest_path, rows=len(row_manifest)),
        "external_panel": _binding(panel, rows=len(rows)),
        "prompt_contract": {
            "training_config": str(training_config),
            "training_config_sha256": config_sha,
            "system_prompt_sha256": native_probe.common_probe.sha256_text(system_prompt),
            "user_prompt_suffix_sha256": (
                native_probe.common_probe.sha256_text(suffix)
                if suffix is not None
                else None
            ),
        },
        "tokenizer": native_probe._tokenizer_fingerprint(tokenizer_path),
        "selection": {
            "algorithm": "external-reference-length-tertiles-domain-hash-v1",
            "selection_domain": native_eval.TASK_CONTRACT_ID,
            "ordering": "(reference_minutes_tokens,sample_id)",
            "bucket_boundaries": list(boundaries),
            "within_bucket_rank": (
                "sha256(selection_domain|release_manifest_sha256|sample_id)"
            ),
            "samples_per_bucket": 4,
            "rows": len(selected),
            "buckets": ["short", "medium", "long"],
            "checkpoint_selection_split": "none_external_evaluation_only",
            "evaluation_split": "external_holdout_1993_2008",
        },
        "samples": selected,
        "limitations": {
            "smoke_only": True,
            "statistical_inference_authorized": False,
            "official_minutes_used_as_model_input": False,
            "reference_type": "deterministic_source_grounded_minutes_style_v1",
        },
    }
    sample_path = output_dir / "samples_n12.json"
    _write_new_json(sample_path, seal_manifest(sample_payload))
    # Invoke the production loader now so a preparation can never publish a
    # manifest that the inference runner will later interpret differently.
    native_eval._load_sample_manifest(sample_path, sha256_file(sample_path))
    native_eval._load_bound_rows(_read_json(sample_path))

    preparation = seal_manifest(
        {
            "schema_version": OUTPUT_SCHEMA,
            "status": "complete",
            "release": {
                "path": str(release_manifest),
                "sha256": release_sha,
                "payload_sha256": release_payload_sha,
            },
            "expanded_rows": len(rows),
            "selected_rows": len(selected),
            "selected_meetings": len(
                {str(row["meeting_start_date"]) for row in selected}
            ),
            "selected_topics": len({str(row["topic"]) for row in selected}),
            "sample_manifest": _binding(sample_path),
            "compatibility_dataset": _binding(
                dataset_path, rows=len(compatibility_rows)
            ),
            "compatibility_row_manifest": _binding(
                row_manifest_path, rows=len(row_manifest)
            ),
        }
    )
    _write_new_json(output_dir / "preparation.json", preparation)
    return preparation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = prepare(
            release_manifest=args.release_manifest,
            panel=args.panel,
            tokenizer_path=args.tokenizer,
            training_config=args.training_config,
            output_dir=args.output_dir,
        )
    except (ExternalSmokeError, OSError, ValueError) as exc:
        print(_canonical_json({"status": "failed", "error": str(exc)}))
        return 1
    print(
        _canonical_json(
            {
                "status": result["status"],
                "payload_sha256": result["integrity"]["payload_sha256"],
                "sample_manifest": result["sample_manifest"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
