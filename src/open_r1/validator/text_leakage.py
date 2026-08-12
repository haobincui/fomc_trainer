"""Deterministic prompt/reference leakage guards.

The common-test checkpoint evaluation treats a shared run of 20 normalized
tokens as evidence that reference text entered a generation prompt.  The
threshold is intentionally long enough not to reject ordinary short phrases.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


PROMPT_REFERENCE_OVERLAP_TOKEN_COUNT = 20
_NORMALIZED_TOKEN_RE = re.compile(
    r"[^\W_]+(?:'[^\W_]+)*",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class ContiguousTokenOverlap:
    """Location of one normalized token-window overlap."""

    token_count: int
    prompt_token_start: int
    reference_token_start: int


def normalized_leakage_tokens(text: str) -> tuple[str, ...]:
    """Tokenize text after deterministic Unicode and case normalization.

    Punctuation and whitespace differences do not hide copied prose. Unicode
    format controls (for example zero-width joiners) are removed before
    tokenization so they cannot be inserted to evade the guard.
    """

    if not isinstance(text, str):
        raise TypeError("Leakage validation requires string prompt/reference text")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    ).replace("\u2019", "'")
    return tuple(_NORMALIZED_TOKEN_RE.findall(normalized))


def find_prompt_reference_token_overlap(
    prompt: str,
    reference: str,
) -> ContiguousTokenOverlap | None:
    """Return the first shared normalized 20-token window, if one exists."""

    window_size = PROMPT_REFERENCE_OVERLAP_TOKEN_COUNT
    prompt_tokens = normalized_leakage_tokens(prompt)
    reference_tokens = normalized_leakage_tokens(reference)
    if len(prompt_tokens) < window_size or len(reference_tokens) < window_size:
        return None

    prompt_windows: dict[tuple[str, ...], int] = {}
    for start in range(len(prompt_tokens) - window_size + 1):
        window = prompt_tokens[start : start + window_size]
        prompt_windows.setdefault(window, start)
    for reference_start in range(len(reference_tokens) - window_size + 1):
        window = reference_tokens[
            reference_start : reference_start + window_size
        ]
        prompt_start = prompt_windows.get(window)
        if prompt_start is not None:
            return ContiguousTokenOverlap(
                token_count=window_size,
                prompt_token_start=prompt_start,
                reference_token_start=reference_start,
            )
    return None


def validate_no_prompt_reference_token_overlap(
    prompt: str,
    reference: str,
    *,
    sample_id: str,
) -> None:
    """Fail closed when a prompt shares 20 consecutive tokens with its reference."""

    overlap = find_prompt_reference_token_overlap(prompt, reference)
    if overlap is None:
        return
    raise ValueError(
        "Normalized contiguous "
        f"{overlap.token_count}-token reference leakage detected in prompt "
        f"{sample_id!r} (prompt token {overlap.prompt_token_start}, "
        f"reference token {overlap.reference_token_start})"
    )
