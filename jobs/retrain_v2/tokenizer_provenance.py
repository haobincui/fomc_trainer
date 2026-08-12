"""Deterministic provenance for the local tokenizer/config dependency bundle."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from jobs.retrain_v2.chk1.contracts import canonical_json, sha256_text
from open_r1.provenance import sha256_file


TOKENIZER_BUNDLE_SCHEMA_VERSION = 1
TOKENIZER_LOADER_CONTRACT = {
    "loader": "transformers.AutoTokenizer.from_pretrained",
    "local_files_only": True,
    "trust_remote_code": False,
    "use_fast": True,
}

# These are the files the pinned local AutoTokenizer is permitted to resolve.
# Keeping this explicit prevents an unbound chat template or token map from being
# silently picked up after the release was audited.
TOKENIZER_ROOT_FILES = frozenset(
    {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "tokenizer.model",
        "sentencepiece.bpe.model",
        "spiece.model",
        "chat_template.jinja",
    }
)


class TokenizerBundleError(ValueError):
    """Raised when the local tokenizer dependency closure is not immutable."""


def _artifact_sha(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record['path']}\0{record['bytes']}\0{record['sha256']}\n".encode()
        )
    return digest.hexdigest()


def snapshot_tokenizer_bundle(model_dir: str | Path) -> dict[str, Any]:
    """Fingerprint every file the fixed local AutoTokenizer may consume."""

    root = Path(model_dir).resolve()
    if not root.is_dir() or root.is_symlink():
        raise TokenizerBundleError(f"tokenizer model directory is invalid: {root}")
    candidates: list[Path] = []
    for name in sorted(TOKENIZER_ROOT_FILES):
        lexical = root / name
        if lexical.is_symlink():
            raise TokenizerBundleError(f"tokenizer bundle contains symlink: {lexical}")
        if lexical.is_file():
            candidates.append(lexical)
    templates = root / "chat_templates"
    if templates.is_symlink():
        raise TokenizerBundleError(f"tokenizer bundle contains symlink: {templates}")
    if templates.exists():
        if not templates.is_dir():
            raise TokenizerBundleError("chat_templates must be a directory")
        for candidate in sorted(templates.rglob("*")):
            if candidate.is_symlink():
                raise TokenizerBundleError(
                    f"tokenizer bundle contains symlink: {candidate}"
                )
            if candidate.is_file():
                if candidate.suffix != ".jinja":
                    raise TokenizerBundleError(
                        f"unexpected tokenizer chat-template artifact: {candidate}"
                    )
                candidates.append(candidate)
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    observed = {path.relative_to(root).as_posix() for path in candidates}
    missing = sorted(required - observed)
    if missing:
        raise TokenizerBundleError(
            f"tokenizer bundle is missing required file(s): {missing}"
        )
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(candidates)
    ]
    payload = {
        "schema_version": TOKENIZER_BUNDLE_SCHEMA_VERSION,
        "loader_contract": dict(TOKENIZER_LOADER_CONTRACT),
        "files": records,
    }
    return {
        **payload,
        "artifact_sha256": _artifact_sha(records),
        "payload_sha256": sha256_text(canonical_json(payload)),
        "file_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
    }


def materialize_tokenizer_bundle(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    expected_snapshot: dict[str, Any],
) -> None:
    """Copy exactly the snapshotted dependency closure with exclusive files."""

    source = Path(source_dir).resolve()
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=False)
    for record in expected_snapshot["files"]:
        relative = Path(record["path"])
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (source / relative).read_bytes()
        if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise TokenizerBundleError(
                f"tokenizer source changed while copying: {relative.as_posix()}"
            )
        with target.open("xb") as handle:
            handle.write(payload)
            handle.flush()
    observed = snapshot_tokenizer_bundle(destination)
    if observed != expected_snapshot:
        raise TokenizerBundleError("materialized tokenizer bundle fingerprint mismatch")
