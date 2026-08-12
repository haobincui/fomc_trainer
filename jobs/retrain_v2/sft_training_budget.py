"""Training-only token contract for the chk1 SFT release.

The completed DeepSeek acquisition bundle records the historical
3072/1024/4096 observation in every row.  That record is immutable provenance,
not the training admission policy.  Training keeps the prompt bound, removes
the standalone completion ceiling, and requires the complete sample to fit the
7168-token GPU1-tested context without truncation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from jobs.retrain_v2.chk1.prompt_projection import (
    ANALYSIS_SFT_SYSTEM_PROMPT,
    PromptProjectionError,
)
from open_r1.trainer.sft_prompt_renderer import (
    SftPromptRenderError,
    render_sft_prompt,
    tokenize_sft_text,
)


SFT_TRAINING_PROMPT_TOKEN_LIMIT = 3072
SFT_TRAINING_COMPLETION_TOKEN_LIMIT: None = None
SFT_TRAINING_TOTAL_TOKEN_LIMIT = 7168
SFT_TRAINING_SCHEMA_VERSION = "chk1-sft-training-token-budget-v3"


class SftTrainingTokenAuditor(Protocol):
    def __call__(self, prompt: str, response: str) -> Mapping[str, Any]: ...


def build_training_sft_token_auditor(tokenizer: Any) -> SftTrainingTokenAuditor:
    """Count one complete SFT sample under the training admission contract."""

    def audit(prompt: str, response: str) -> Mapping[str, Any]:
        try:
            rendered = render_sft_prompt(
                tokenizer,
                [
                    {"role": "system", "content": ANALYSIS_SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except SftPromptRenderError as exc:
            raise PromptProjectionError(str(exc)) from exc

        completion = response
        eos = getattr(tokenizer, "eos_token", None)
        if isinstance(eos, str) and eos and not completion.endswith(eos):
            completion += eos
        try:
            prompt_ids = tokenize_sft_text(tokenizer, rendered)
            full_ids = tokenize_sft_text(tokenizer, rendered + completion)
        except SftPromptRenderError as exc:
            raise PromptProjectionError(str(exc)) from exc
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise PromptProjectionError("SFT tokenizer continuation prefix mismatch")
        completion_tokens = len(full_ids) - len(prompt_ids)
        payload = {
            "schema_version": SFT_TRAINING_SCHEMA_VERSION,
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": completion_tokens,
            "total_tokens": len(full_ids),
            "max_prompt_tokens": SFT_TRAINING_PROMPT_TOKEN_LIMIT,
            "max_completion_tokens": SFT_TRAINING_COMPLETION_TOKEN_LIMIT,
            "max_total_tokens": SFT_TRAINING_TOTAL_TOKEN_LIMIT,
            "overflow_policy": "error",
            "truncated": False,
        }
        payload["passed"] = (
            payload["prompt_tokens"] <= SFT_TRAINING_PROMPT_TOKEN_LIMIT
            and payload["total_tokens"] <= SFT_TRAINING_TOTAL_TOKEN_LIMIT
        )
        return payload

    return audit


__all__ = [
    "SFT_TRAINING_COMPLETION_TOKEN_LIMIT",
    "SFT_TRAINING_PROMPT_TOKEN_LIMIT",
    "SFT_TRAINING_SCHEMA_VERSION",
    "SFT_TRAINING_TOTAL_TOKEN_LIMIT",
    "build_training_sft_token_auditor",
]
