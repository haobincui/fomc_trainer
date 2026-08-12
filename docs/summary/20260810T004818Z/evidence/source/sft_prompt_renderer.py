"""Shared prompt rendering contract for prompt-completion SFT datasets.

TRL tokenizes non-conversational prompt/completion strings with the tokenizer's
default ``add_special_tokens`` behavior. Some chat templates (including the
DeepSeek-R1-Distill-Llama tokenizer used by chk1) already render a literal BOS
token. Passing that rendered string to TRL unchanged therefore creates two BOS
tokens. This module removes the one template-owned literal BOS and proves that
normal string tokenization is identical to the tokenizer's canonical
``apply_chat_template(tokenize=True)`` result.

The completion intentionally remains a separate raw string. Converting the
full target into an assistant chat message is not equivalent for the DeepSeek
template because it removes the reasoning prefix before ``</think>``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class SftPromptRenderError(ValueError):
    """Raised when a chat prompt cannot be rendered with exact token parity."""


def _as_token_ids(value: Any, *, source: str) -> list[int]:
    """Normalize common tokenizer outputs to one flat list of token IDs."""

    if isinstance(value, Mapping):
        value = value.get("input_ids")
    elif hasattr(value, "input_ids"):
        value = value.input_ids

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SftPromptRenderError(f"{source} returned invalid input_ids")
    if value and isinstance(value[0], Sequence) and not isinstance(
        value[0], (str, bytes)
    ):
        if len(value) != 1:
            raise SftPromptRenderError(f"{source} returned an unexpected batch")
        value = value[0]
    try:
        return [int(token_id) for token_id in value]
    except (TypeError, ValueError) as exc:
        raise SftPromptRenderError(
            f"{source} returned non-integer input_ids"
        ) from exc


def tokenize_sft_text(tokenizer: Any, text: str) -> list[int]:
    """Tokenize one TRL prompt/completion string with default special tokens."""

    if not isinstance(text, str):
        raise SftPromptRenderError("SFT text must be a string")
    try:
        encoded = tokenizer(text=text)
    except Exception as exc:  # pragma: no cover - tokenizer-specific detail
        raise SftPromptRenderError(f"SFT text tokenization failed: {exc}") from exc
    return _as_token_ids(encoded, source="SFT text tokenizer")


def render_sft_prompt(tokenizer: Any, messages: Sequence[Mapping[str, Any]]) -> str:
    """Render a prompt string that TRL will tokenize with exactly one BOS.

    The returned string excludes the literal BOS owned by the chat template.
    Encoding the result through the tokenizer's normal ``__call__`` path must
    produce exactly the same IDs as ``apply_chat_template(tokenize=True)``.
    Any mismatch is rejected instead of silently training on a different chat
    boundary.
    """

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise SftPromptRenderError("tokenizer lacks apply_chat_template")
    bos_token = getattr(tokenizer, "bos_token", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    has_bos_token = isinstance(bos_token, str) and bool(bos_token)
    has_bos_token_id = bos_token_id is not None
    if has_bos_token != has_bos_token_id:
        raise SftPromptRenderError(
            "tokenizer has inconsistent bos_token and bos_token_id"
        )
    if has_bos_token_id:
        try:
            bos_token_id = int(bos_token_id)
        except (TypeError, ValueError) as exc:
            raise SftPromptRenderError("tokenizer has an invalid bos_token_id") from exc

    try:
        rendered = apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        canonical_ids = _as_token_ids(
            apply_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                truncation=False,
                return_dict=False,
            ),
            source="SFT chat template",
        )
    except SftPromptRenderError:
        raise
    except Exception as exc:  # pragma: no cover - tokenizer-specific detail
        raise SftPromptRenderError(f"SFT chat rendering failed: {exc}") from exc

    if not isinstance(rendered, str) or not rendered:
        raise SftPromptRenderError("SFT chat template did not return text")
    normalized = rendered
    if has_bos_token:
        if not rendered.startswith(bos_token):
            raise SftPromptRenderError(
                "SFT chat template text does not start with the tokenizer BOS"
            )

        # Remove exactly the template-owned BOS. A second literal BOS would
        # still be duplicated by default special-token handling, so reject it.
        normalized = rendered[len(bos_token) :]
        if normalized.startswith(bos_token):
            raise SftPromptRenderError(
                "SFT chat template text contains more than one leading BOS"
            )
    if not normalized:
        raise SftPromptRenderError("SFT chat prompt is empty after BOS removal")

    trl_ids = tokenize_sft_text(tokenizer, normalized)
    if trl_ids != canonical_ids:
        raise SftPromptRenderError(
            "TRL string tokenization does not match "
            "apply_chat_template(tokenize=True)"
        )
    if has_bos_token:
        if not trl_ids or trl_ids[0] != bos_token_id:
            raise SftPromptRenderError(
                "SFT prompt does not start with exactly one BOS"
            )
        if trl_ids.count(bos_token_id) != 1:
            raise SftPromptRenderError("SFT prompt contains multiple BOS token IDs")
    return normalized


__all__ = [
    "SftPromptRenderError",
    "render_sft_prompt",
    "tokenize_sft_text",
]
