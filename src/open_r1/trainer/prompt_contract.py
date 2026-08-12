"""Shared prompt composition for training and its token-budget audit."""

from __future__ import annotations


def compose_user_prompt(user_prompt: str, suffix: str | None) -> str:
    """Append an optional runtime contract exactly once and without ambiguity."""

    if not isinstance(user_prompt, str) or not user_prompt:
        raise ValueError("user prompt must be a non-empty string")
    if suffix is None:
        return user_prompt
    if not isinstance(suffix, str) or not suffix.strip():
        raise ValueError("user prompt suffix must be null or a non-empty string")
    return f"{user_prompt.rstrip()}\n\n{suffix.strip()}"
