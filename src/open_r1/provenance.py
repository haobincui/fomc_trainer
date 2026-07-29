"""Immutable artifact fingerprints used by canonical experiment manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_artifact_path(path: str | Path) -> dict:
    """Fingerprint a file or a complete directory tree.

    Directory fingerprints hash a canonical inventory containing each relative
    path, byte size, and file-content SHA-256. This binds a model checkpoint,
    tokenizer, and configuration files together without depending on mtimes.
    """

    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact path must exist locally so it can be fingerprinted: "
            f"{artifact_path}"
        )

    if artifact_path.is_file():
        return {
            "path": str(artifact_path),
            "kind": "file",
            "sha256": sha256_file(artifact_path),
            "file_count": 1,
            "total_bytes": artifact_path.stat().st_size,
            "algorithm": "sha256(file_bytes)",
        }

    files = sorted(
        candidate
        for candidate in artifact_path.rglob("*")
        if candidate.is_file()
    )
    if not files:
        raise ValueError(f"Artifact directory contains no files: {artifact_path}")

    inventory_digest = hashlib.sha256()
    total_bytes = 0
    for candidate in files:
        relative_path = candidate.relative_to(artifact_path).as_posix()
        size = candidate.stat().st_size
        file_digest = sha256_file(candidate)
        total_bytes += size
        inventory_digest.update(
            f"{relative_path}\0{size}\0{file_digest}\n".encode("utf-8")
        )

    return {
        "path": str(artifact_path),
        "kind": "directory",
        "sha256": inventory_digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "algorithm": (
            "sha256(sorted UTF-8 records "
            "'<relative_path>\\0<size>\\0<file_sha256>\\n')"
        ),
    }


def validate_sha256(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256 digest")
    return text
