"""Create a local-model semantic screen for the official-Minutes candidate.

This script never promotes the source release to training-ready status. It
uses a locally served Ollama judge to identify a provisional subset for human
review, while preserving the deterministic source release and every judge
response. The screen asks whether the official paragraph is supported by the
analysis, whether both texts address the same specific topic, and whether the
legacy reasoning is compatible with the official final answer.

The source analyses are reference-conditioned and are not bound to the
canonical D-1 evidence replay. A machine PASS does not repair either issue and
does not substitute for sentence-level human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / (
    "dataset/processed/retrain_v2/chk2_official_minutes_v1_20260826"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "dataset/processed/retrain_v2/"
    "chk2_official_minutes_semantic_qwen3_14b_v1_20260826"
)
DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:14b"
DEFAULT_BATCH_SIZE = 4
DEFAULT_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_SEED = 20260826
SPLITS = ("train", "validation", "test")
SCHEMA_VERSION = "chk2-official-minutes-machine-semantic-release-v1"
ROW_SCHEMA_VERSION = "chk2-official-minutes-machine-semantic-row-v1"

JUDGE_SYSTEM_PROMPT = """\
You are a strict semantic auditor for analysis-to-official-FOMC-Minutes SFT.
Evaluate every supplied pair independently.

The official target may omit any source detail. Omission alone is not an
error. Set an axis to true only when all of its requirements are satisfied:

1. target_supported: Every factual claim in the official target is entailed
   by the analysis without outside knowledge. Reject contradictions, changed
   direction or timing, added entities, added causes, and unsupported details.
2. same_topic: The target and analysis address the same specific economic or
   financial phenomenon. Generic macroeconomic or financial overlap is not
   enough.
3. reasoning_compatible: The reasoning is grounded in the analysis and is
   logically compatible with the official target actually produced. Reject a
   plan that promises facts, quantities, directions, or coverage contradicted
   by the final target.

Confidence is confidence in the complete three-axis verdict, not the
probability that a row passes. Return JSON only in this exact shape:
{"audits":[{"sample_id":"...","target_supported":true,
"same_topic":true,"reasoning_compatible":true,"confidence":0.95,
"note":"under 35 words"}]}
Return one audit for every input ID, in the same order.\
"""


class SemanticAuditError(RuntimeError):
    """The local semantic audit could not be completed reproducibly."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write(path, "".join(_canonical_json(dict(row)) + "\n" for row in rows))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticAuditError(f"cannot read JSON object: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SemanticAuditError(f"JSON value is not an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SemanticAuditError(f"cannot read JSONL file: {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise SemanticAuditError(f"blank JSONL row: {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SemanticAuditError(
                f"invalid JSONL row: {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise SemanticAuditError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def _post_json(url: str, payload: Mapping[str, Any], *, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SemanticAuditError(f"local judge request failed: {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise SemanticAuditError("local judge response is not a JSON object")
    return value


def _get_json(url: str, *, timeout: int) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            value = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SemanticAuditError(f"local model registry request failed: {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise SemanticAuditError("local model registry response is not an object")
    return value


def _model_record(endpoint: str, model: str, *, timeout: int) -> dict[str, Any]:
    registry = _get_json(f"{endpoint.rstrip('/')}/api/tags", timeout=timeout)
    models = registry.get("models")
    if not isinstance(models, list):
        raise SemanticAuditError("local model registry lacks a models list")
    for candidate in models:
        if not isinstance(candidate, dict):
            continue
        names = {str(candidate.get("name") or ""), str(candidate.get("model") or "")}
        if model in names:
            digest = str(candidate.get("digest") or "")
            if len(digest) != 64:
                raise SemanticAuditError(f"local model digest is invalid: {model}")
            return {
                "name": model,
                "digest": digest,
                "size": candidate.get("size"),
                "modified_at": candidate.get("modified_at"),
            }
    raise SemanticAuditError(f"local judge model is not installed: {model}")


def _validate_verdicts(
    content: str, *, expected_ids: Sequence[str]
) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SemanticAuditError(f"judge content is not JSON: {exc}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("audits"), list):
        raise SemanticAuditError("judge content lacks an audits list")
    audits = parsed["audits"]
    if len(audits) != len(expected_ids):
        raise SemanticAuditError(
            f"judge returned {len(audits)} audits for {len(expected_ids)} rows"
        )
    validated: list[dict[str, Any]] = []
    for expected_id, audit in zip(expected_ids, audits, strict=True):
        if not isinstance(audit, dict) or audit.get("sample_id") != expected_id:
            raise SemanticAuditError(f"judge sample order or identity changed: {expected_id}")
        for field in ("target_supported", "same_topic", "reasoning_compatible"):
            if not isinstance(audit.get(field), bool):
                raise SemanticAuditError(f"judge field is not boolean: {expected_id}.{field}")
        confidence = audit.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
        ):
            raise SemanticAuditError(f"judge confidence is invalid: {expected_id}")
        note = audit.get("note")
        if not isinstance(note, str) or not note.strip() or len(note) > 500:
            raise SemanticAuditError(f"judge note is invalid: {expected_id}")
        validated.append(
            {
                "sample_id": expected_id,
                "target_supported": audit["target_supported"],
                "same_topic": audit["same_topic"],
                "reasoning_compatible": audit["reasoning_compatible"],
                "confidence": round(float(confidence), 6),
                "note": note.strip(),
            }
        )
    return validated


def _failure_reasons(
    verdict: Mapping[str, Any], *, confidence_threshold: float
) -> list[str]:
    reasons = [
        field
        for field in ("target_supported", "same_topic", "reasoning_compatible")
        if verdict.get(field) is not True
    ]
    if float(verdict.get("confidence", 0)) < confidence_threshold:
        reasons.append("confidence_below_threshold")
    return reasons


def _judge_once(
    pairs: Sequence[Mapping[str, str]],
    *,
    endpoint: str,
    model: str,
    timeout: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    user_payload = {"pairs": [dict(pair) for pair in pairs]}
    request_payload = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": "60m",
        "format": "json",
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_predict": max(256, len(pairs) * 180),
        },
    }
    raw = _post_json(
        f"{endpoint.rstrip('/')}/api/chat",
        request_payload,
        timeout=timeout,
    )
    message = raw.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise SemanticAuditError("local judge returned empty message content")
    expected_ids = [str(pair["sample_id"]) for pair in pairs]
    verdicts = _validate_verdicts(content, expected_ids=expected_ids)
    batch_record = {
        "sample_ids": expected_ids,
        "request_sha256": _sha256_text(_canonical_json(request_payload)),
        "response_content": content,
        "response_content_sha256": _sha256_text(content),
        "done_reason": raw.get("done_reason"),
        "load_duration_ns": raw.get("load_duration"),
        "prompt_eval_count": raw.get("prompt_eval_count"),
        "prompt_eval_duration_ns": raw.get("prompt_eval_duration"),
        "eval_count": raw.get("eval_count"),
        "eval_duration_ns": raw.get("eval_duration"),
        "total_duration_ns": raw.get("total_duration"),
    }
    return verdicts, batch_record


def _judge_with_fallback(
    pairs: Sequence[Mapping[str, str]],
    *,
    endpoint: str,
    model: str,
    timeout: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        verdicts, batch = _judge_once(
            pairs,
            endpoint=endpoint,
            model=model,
            timeout=timeout,
            seed=seed,
        )
        batch["fallback"] = False
        return verdicts, [batch]
    except SemanticAuditError:
        if len(pairs) == 1:
            raise
    verdicts = []
    batches = []
    for pair in pairs:
        single_verdicts, batch = _judge_once(
            [pair],
            endpoint=endpoint,
            model=model,
            timeout=timeout,
            seed=seed,
        )
        batch["fallback"] = True
        verdicts.extend(single_verdicts)
        batches.append(batch)
    return verdicts, batches


def _file_record(root: Path, relative: str, *, rows: int | None = None) -> dict[str, Any]:
    path = root / relative
    record: dict[str, Any] = {
        "path": relative,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _meeting_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    meetings = sorted({str(row["meeting_date"]) for row in rows})
    return {
        "meetings": len(meetings),
        "first_meeting": meetings[0] if meetings else None,
        "last_meeting": meetings[-1] if meetings else None,
    }


def _load_source_release(
    source_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    release = _load_json(source_root / "release_manifest.json")
    handoff = _load_json(source_root / "handoff.json")
    expected_release_sha = handoff.get("release_manifest", {}).get("sha256")
    if expected_release_sha != _sha256_file(source_root / "release_manifest.json"):
        raise SemanticAuditError("source handoff does not bind the release manifest")
    if release.get("analysis_is_reference_free") is not False:
        raise SemanticAuditError("source release lost its reference-conditioned warning")
    counts = release.get("split_counts")
    if not isinstance(counts, dict) or set(counts) != set(SPLITS):
        raise SemanticAuditError("source release split contract is invalid")

    data_by_split: dict[str, list[dict[str, Any]]] = {}
    manifests_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        data_relative = f"minutes_alignment/{split}.jsonl"
        manifest_relative = f"minutes_alignment/manifests/{split}.jsonl"
        for relative in (data_relative, manifest_relative):
            descriptor = release.get("files", {}).get(relative)
            if not isinstance(descriptor, dict):
                raise SemanticAuditError(f"source release lacks file record: {relative}")
            path = source_root / relative
            if (
                _sha256_file(path) != descriptor.get("sha256")
                or path.stat().st_size != descriptor.get("bytes")
            ):
                raise SemanticAuditError(f"source release file changed: {relative}")
        data_rows = _load_jsonl(source_root / data_relative)
        manifests = _load_jsonl(source_root / manifest_relative)
        expected = counts[split]
        if len(data_rows) != expected or len(manifests) != expected:
            raise SemanticAuditError(f"source row count changed: {split}")
        for index, (data, manifest) in enumerate(
            zip(data_rows, manifests, strict=True), 1
        ):
            if set(data) != {"prompt", "response"}:
                raise SemanticAuditError(f"source SFT schema changed: {split}:{index}")
            if (
                _sha256_text(str(data["prompt"])) != manifest.get("prompt_sha256")
                or _sha256_text(str(data["response"]))
                != manifest.get("response_sha256")
            ):
                raise SemanticAuditError(f"source SFT hash mismatch: {split}:{index}")
        data_by_split[split] = data_rows
        manifests_by_split[split] = manifests
    return release, data_by_split, manifests_by_split


def audit_release(
    *,
    source_root: Path,
    output_root: Path,
    endpoint: str,
    model: str,
    batch_size: int,
    confidence_threshold: float,
    timeout: int,
    seed: int,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise SemanticAuditError(f"output release already exists: {output_root}")
    if batch_size <= 0:
        raise SemanticAuditError("batch size must be positive")
    if not 0 <= confidence_threshold <= 1:
        raise SemanticAuditError("confidence threshold must lie in [0, 1]")

    source_release, data_by_split, manifests_by_split = _load_source_release(source_root)
    model_record = _model_record(endpoint, model, timeout=timeout)
    staging_parent = output_root.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=staging_parent))
    try:
        passed_data: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
        passed_manifests: dict[str, list[dict[str, Any]]] = {
            split: [] for split in SPLITS
        }
        audits_by_split: dict[str, list[dict[str, Any]]] = {
            split: [] for split in SPLITS
        }
        rejected_by_split: dict[str, list[dict[str, Any]]] = {
            split: [] for split in SPLITS
        }
        judge_batches: list[dict[str, Any]] = []

        for split in SPLITS:
            manifests = manifests_by_split[split]
            for offset in range(0, len(manifests), batch_size):
                batch_manifests = manifests[offset : offset + batch_size]
                pairs = [
                    {
                        "sample_id": str(row["sample_id"]),
                        "analysis": str(row["analysis"]),
                        "reasoning": str(row["reasoning"]),
                        "official_target": str(row["official_minutes_paragraph"]),
                    }
                    for row in batch_manifests
                ]
                verdicts, batches = _judge_with_fallback(
                    pairs,
                    endpoint=endpoint,
                    model=model,
                    timeout=timeout,
                    seed=seed,
                )
                for batch in batches:
                    batch["batch_index"] = len(judge_batches)
                    batch["split"] = split
                    judge_batches.append(batch)
                for local_index, (manifest, verdict) in enumerate(
                    zip(batch_manifests, verdicts, strict=True)
                ):
                    source_index = offset + local_index
                    failure_reasons = _failure_reasons(
                        verdict,
                        confidence_threshold=confidence_threshold,
                    )
                    machine_pass = not failure_reasons
                    record = {
                        "schema_version": ROW_SCHEMA_VERSION,
                        "sample_id": manifest["sample_id"],
                        "split": split,
                        "meeting_date": manifest["meeting_date"],
                        "topic": manifest["topic"],
                        "source_row_index": manifest["source_row_index"],
                        "analysis_sha256": manifest["analysis_sha256"],
                        "reasoning_sha256": manifest["reasoning_sha256"],
                        "official_minutes_sha256": manifest[
                            "official_minutes_sha256"
                        ],
                        "source_response_sha256": manifest["response_sha256"],
                        "judge": verdict,
                        "machine_pass": machine_pass,
                        "failure_reasons": failure_reasons,
                        "human_review_status": "not_run",
                    }
                    audits_by_split[split].append(record)
                    if machine_pass:
                        projected_manifest = dict(manifest)
                        projected_manifest["semantic_screen"] = record
                        projected_manifest["training_readiness"] = (
                            "human_review_required"
                        )
                        passed_data[split].append(data_by_split[split][source_index])
                        passed_manifests[split].append(projected_manifest)
                    else:
                        rejected_by_split[split].append(record)
                print(
                    f"semantic audit {split}: {min(offset + batch_size, len(manifests))}"
                    f"/{len(manifests)}",
                    flush=True,
                )

            _write_jsonl(
                staging / f"minutes_alignment/{split}.jsonl",
                passed_data[split],
            )
            _write_jsonl(
                staging / f"minutes_alignment/manifests/{split}.jsonl",
                passed_manifests[split],
            )
            _write_jsonl(
                staging / f"semantic_audits/{split}.jsonl",
                audits_by_split[split],
            )
            _write_jsonl(
                staging / f"semantic_rejections/{split}.jsonl",
                rejected_by_split[split],
            )

        _write_jsonl(staging / "semantic_audits/judge_batches.jsonl", judge_batches)
        all_audits = [row for split in SPLITS for row in audits_by_split[split]]
        all_passed = [row for split in SPLITS for row in passed_manifests[split]]
        pass_counts = {split: len(passed_data[split]) for split in SPLITS}
        input_counts = {split: len(data_by_split[split]) for split in SPLITS}
        rejection_counts = {
            split: len(rejected_by_split[split]) for split in SPLITS
        }
        axis_failure_counts = Counter(
            reason for row in all_audits for reason in row["failure_reasons"]
        )
        semantic_audit = {
            "schema_version": SCHEMA_VERSION,
            "status": "local_machine_screen_complete_human_review_pending",
            "paper_model": "chk-2 official-target candidate variant",
            "source_release_id": source_release["release_id"],
            "source_release_manifest_sha256": _sha256_file(
                source_root / "release_manifest.json"
            ),
            "input_counts": input_counts,
            "machine_pass_counts": pass_counts,
            "machine_rejected_counts": rejection_counts,
            "total_input_rows": len(all_audits),
            "total_machine_pass_rows": len(all_passed),
            "total_machine_rejected_rows": sum(rejection_counts.values()),
            "machine_pass_fraction": round(
                len(all_passed) / len(all_audits), 8
            ),
            "axis_failure_row_counts": dict(sorted(axis_failure_counts.items())),
            "machine_pass_meetings": {
                split: _meeting_summary(passed_manifests[split]) for split in SPLITS
            },
            "judge": {
                **model_record,
                "endpoint": endpoint,
                "system_prompt_sha256": _sha256_text(JUDGE_SYSTEM_PROMPT),
                "temperature": 0,
                "seed": seed,
                "batch_size": batch_size,
                "confidence_threshold": confidence_threshold,
                "raw_batch_responses_retained": True,
            },
            "human_review": {
                "status": "not_run",
                "training_readiness": False,
                "warning": (
                    "A single local-model screen is not human sentence-level "
                    "validation and may contain false acceptances and rejections."
                ),
            },
            "unresolved_source_limitations": {
                "analysis_reference_conditioned": True,
                "point_in_time_status": "not_established_for_legacy_lineage",
                "reasoning_generated_against_official_target": False,
                "suitable_for_leakage_safe_evaluation": False,
            },
        }
        _write_json(staging / "audits/semantic_quality.json", semantic_audit)
        _write_json(
            staging / "judge_contract.json",
            {
                "schema_version": SCHEMA_VERSION,
                "system_prompt": JUDGE_SYSTEM_PROMPT,
                "system_prompt_sha256": _sha256_text(JUDGE_SYSTEM_PROMPT),
                "model": model_record,
                "endpoint": endpoint,
                "options": {
                    "temperature": 0,
                    "seed": seed,
                    "batch_size": batch_size,
                    "confidence_threshold": confidence_threshold,
                },
            },
        )

        files: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            for relative, rows in (
                (f"minutes_alignment/{split}.jsonl", len(passed_data[split])),
                (
                    f"minutes_alignment/manifests/{split}.jsonl",
                    len(passed_manifests[split]),
                ),
                (f"semantic_audits/{split}.jsonl", len(audits_by_split[split])),
                (
                    f"semantic_rejections/{split}.jsonl",
                    len(rejected_by_split[split]),
                ),
            ):
                files[relative] = _file_record(staging, relative, rows=rows)
        files["semantic_audits/judge_batches.jsonl"] = _file_record(
            staging,
            "semantic_audits/judge_batches.jsonl",
            rows=len(judge_batches),
        )
        files["audits/semantic_quality.json"] = _file_record(
            staging, "audits/semantic_quality.json"
        )
        files["judge_contract.json"] = _file_record(staging, "judge_contract.json")
        release_manifest = {
            "schema_version": SCHEMA_VERSION,
            "release_id": output_root.name,
            "created_at_utc": _utc_now(),
            "paper_model": "chk-2 official-target candidate variant",
            "dataset_role": "machine_semantic_screen_for_human_review_only",
            "quality_status": "machine_screen_complete_human_review_pending",
            "training_ready": False,
            "reported_checkpoint_eligible": False,
            "immutable_inputs": True,
            "dataset_path": _display_path(output_root / "minutes_alignment"),
            "source": {
                "release_id": source_release["release_id"],
                "path": _display_path(source_root),
                "release_manifest_sha256": _sha256_file(
                    source_root / "release_manifest.json"
                ),
            },
            "judge": model_record,
            "split_counts": pass_counts,
            "total_rows": len(all_passed),
            "builder": {
                "path": _display_path(Path(__file__)),
                "sha256": _sha256_file(Path(__file__)),
            },
            "files": files,
        }
        _write_json(staging / "release_manifest.json", release_manifest)
        handoff = {
            "schema_version": SCHEMA_VERSION,
            "release_id": output_root.name,
            "paper_model": release_manifest["paper_model"],
            "dataset_path": release_manifest["dataset_path"],
            "quality_status": release_manifest["quality_status"],
            "training_ready": False,
            "human_review_status": "not_run",
            "split_counts": pass_counts,
            "total_rows": len(all_passed),
            "release_manifest": {
                "path": "release_manifest.json",
                "sha256": _sha256_file(staging / "release_manifest.json"),
            },
            "semantic_quality_audit": {
                "path": "audits/semantic_quality.json",
                "sha256": _sha256_file(staging / "audits/semantic_quality.json"),
            },
        }
        _write_json(staging / "handoff.json", handoff)
        os.replace(staging, output_root)
        return handoff
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handoff = audit_release(
        source_root=args.source_root,
        output_root=args.output_root,
        endpoint=args.endpoint,
        model=args.model,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        timeout=args.timeout,
        seed=args.seed,
    )
    print(json.dumps(handoff, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
