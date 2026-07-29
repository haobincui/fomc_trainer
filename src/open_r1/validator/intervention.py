"""Validation primitives for indicator-block deletion interventions."""

from __future__ import annotations

import re


def single_contiguous_deletion(
    full_prompt: str,
    masked_prompt: str,
) -> tuple[int, int, str]:
    """Return the exact one-block deletion transforming full into masked."""

    if full_prompt == masked_prompt:
        raise ValueError("masked prompt is identical to the full prompt")
    if len(masked_prompt) >= len(full_prompt):
        raise ValueError("masked prompt is not shorter than the full prompt")

    prefix_length = 0
    prefix_limit = min(len(full_prompt), len(masked_prompt))
    while (
        prefix_length < prefix_limit
        and full_prompt[prefix_length] == masked_prompt[prefix_length]
    ):
        prefix_length += 1

    suffix_length = 0
    suffix_limit = len(masked_prompt) - prefix_length
    while (
        suffix_length < suffix_limit
        and full_prompt[len(full_prompt) - 1 - suffix_length]
        == masked_prompt[len(masked_prompt) - 1 - suffix_length]
    ):
        suffix_length += 1

    deletion_end = len(full_prompt) - suffix_length
    removed_block = full_prompt[prefix_length:deletion_end]
    reconstructed = full_prompt[:prefix_length] + full_prompt[deletion_end:]
    if not removed_block or reconstructed != masked_prompt:
        raise ValueError(
            "masked prompt differs from the full prompt by more than one "
            "contiguous deletion"
        )
    return prefix_length, deletion_end, removed_block


def normalise_indicator_text(value: object) -> str:
    text = str(value or "").casefold().replace("labour", "labor")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def match_indicator_marker(
    removed_block: str,
    markers: list[str],
) -> str | None:
    semantic_lines = [
        normalise_indicator_text(line)
        for line in removed_block.splitlines()
        if normalise_indicator_text(line)
    ]
    if not semantic_lines:
        return None
    normalised_header = semantic_lines[0]
    for marker in markers:
        normalised_marker = normalise_indicator_text(marker)
        if (
            normalised_marker
            and f" {normalised_marker} " in f" {normalised_header} "
        ):
            return marker
    return None


def has_line_block_boundaries(
    full_prompt: str,
    deletion_start: int,
    deletion_end: int,
) -> bool:
    start_is_boundary = (
        deletion_start == 0
        or full_prompt[deletion_start - 1] == "\n"
        or full_prompt[deletion_start] == "\n"
    )
    end_is_boundary = (
        deletion_end == len(full_prompt)
        or full_prompt[deletion_end] == "\n"
        or full_prompt[deletion_end - 1] == "\n"
    )
    return start_is_boundary and end_is_boundary
