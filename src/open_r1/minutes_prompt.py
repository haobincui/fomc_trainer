"""Frozen system-prompt contract for canonical Minutes generation.

The indicator-analysis prompt deliberately lives elsewhere.  This module is
only for the second-stage model that drafts one requested Minutes section.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MINUTES_SYSTEM_PROMPT_VERSION = "canonical-minutes-section-concise-v1"
MINUTES_SYSTEM_PROMPT_TEXT = (
    "You draft exactly one requested FOMC Minutes section. Return only the "
    "finished Minutes-style prose for that section. Do not reveal analysis or "
    "reasoning, do not emit `<think>` or XML/Markdown wrappers, and do not add "
    "a preamble, headings, bullet lists, or any other section. Synthesize only "
    "the evidence in the user prompt and do not quote indicator blocks "
    "verbatim. Be concise and complete. You must finish the section within "
    "4,096 output tokens. Stop immediately after the requested section."
)
MINUTES_SYSTEM_PROMPT_SHA256 = (
    "8bcd4db511b1c0327bb958839ccae97aa1f4653bbd60f6c9aa69e2520537eb21"
)
MINUTES_REQUESTED_MAX_OUTPUT_TOKENS = 4096
MINUTES_HARD_MAX_NEW_TOKENS = 8192
MINUTES_MAX_MODEL_LEN = 16384
MINUTES_INPUT_TRUNCATION_POLICY = "forbidden"
MINUTES_TOKEN_LIMIT_POLICY = "error"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MinutesPromptSpec:
    """Validated immutable prompt and token-budget policy."""

    version: str
    text: str
    sha256: str
    requested_max_output_tokens: int
    hard_max_new_tokens: int
    max_model_len: int
    input_truncation: str
    token_limit_policy: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "text": self.text,
            "sha256": self.sha256,
            "requested_max_output_tokens": self.requested_max_output_tokens,
            "hard_max_new_tokens": self.hard_max_new_tokens,
            "max_model_len": self.max_model_len,
            "input_truncation": self.input_truncation,
            "token_limit_policy": self.token_limit_policy,
        }


CANONICAL_MINUTES_PROMPT_SPEC = MinutesPromptSpec(
    version=MINUTES_SYSTEM_PROMPT_VERSION,
    text=MINUTES_SYSTEM_PROMPT_TEXT,
    sha256=MINUTES_SYSTEM_PROMPT_SHA256,
    requested_max_output_tokens=MINUTES_REQUESTED_MAX_OUTPUT_TOKENS,
    hard_max_new_tokens=MINUTES_HARD_MAX_NEW_TOKENS,
    max_model_len=MINUTES_MAX_MODEL_LEN,
    input_truncation=MINUTES_INPUT_TRUNCATION_POLICY,
    token_limit_policy=MINUTES_TOKEN_LIMIT_POLICY,
)


def validate_minutes_prompt_spec(
    raw: Mapping[str, Any],
    *,
    require_canonical: bool = True,
) -> MinutesPromptSpec:
    """Validate a serialized prompt contract and optionally pin exact text."""

    if not isinstance(raw, Mapping):
        raise ValueError("minutes_system_prompt must be a JSON object")
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("minutes_system_prompt.text must be a non-empty string")
    observed_sha256 = _sha256_text(text)
    declared_sha256 = str(raw.get("sha256") or "").strip().lower()
    if declared_sha256 != observed_sha256:
        raise ValueError(
            "minutes_system_prompt.sha256 does not match its exact text"
        )

    int_fields: dict[str, int] = {}
    for field in (
        "requested_max_output_tokens",
        "hard_max_new_tokens",
        "max_model_len",
    ):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"minutes_system_prompt.{field} must be positive")
        int_fields[field] = value
    if not (
        int_fields["requested_max_output_tokens"]
        <= int_fields["hard_max_new_tokens"]
        < int_fields["max_model_len"]
    ):
        raise ValueError("minutes_system_prompt token limits are inconsistent")

    spec = MinutesPromptSpec(
        version=str(raw.get("version") or "").strip(),
        text=text,
        sha256=declared_sha256,
        requested_max_output_tokens=int_fields["requested_max_output_tokens"],
        hard_max_new_tokens=int_fields["hard_max_new_tokens"],
        max_model_len=int_fields["max_model_len"],
        input_truncation=str(raw.get("input_truncation") or "").strip(),
        token_limit_policy=str(raw.get("token_limit_policy") or "").strip(),
    )
    if not spec.version:
        raise ValueError("minutes_system_prompt.version must be non-empty")
    if require_canonical and spec != CANONICAL_MINUTES_PROMPT_SPEC:
        mismatches = {
            key: {
                "expected": expected,
                "observed": spec.as_dict().get(key),
            }
            for key, expected in CANONICAL_MINUTES_PROMPT_SPEC.as_dict().items()
            if spec.as_dict().get(key) != expected
        }
        raise ValueError(
            "Minutes prompt contract differs from the frozen canonical policy: "
            f"{mismatches}"
        )
    return spec


def load_minutes_prompt_config(
    config_file: str | Path,
) -> tuple[MinutesPromptSpec, dict[str, Any], dict[str, Any]]:
    """Load the prompt contract plus an immutable source-file binding."""

    path = Path(config_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Minutes prompt config does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Minutes prompt config JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Minutes prompt config must be a JSON object")
    spec = validate_minutes_prompt_spec(payload.get("minutes_system_prompt"))
    binding = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "experiment_id": payload.get("experiment_id"),
    }
    return spec, binding, payload


if _sha256_text(MINUTES_SYSTEM_PROMPT_TEXT) != MINUTES_SYSTEM_PROMPT_SHA256:
    raise RuntimeError("Canonical Minutes system-prompt constant has an invalid hash")


__all__ = [
    "CANONICAL_MINUTES_PROMPT_SPEC",
    "MINUTES_HARD_MAX_NEW_TOKENS",
    "MINUTES_MAX_MODEL_LEN",
    "MINUTES_REQUESTED_MAX_OUTPUT_TOKENS",
    "MINUTES_SYSTEM_PROMPT_SHA256",
    "MINUTES_SYSTEM_PROMPT_TEXT",
    "MINUTES_SYSTEM_PROMPT_VERSION",
    "MinutesPromptSpec",
    "load_minutes_prompt_config",
    "validate_minutes_prompt_spec",
]
