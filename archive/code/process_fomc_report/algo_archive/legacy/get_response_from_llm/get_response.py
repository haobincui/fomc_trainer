from __future__ import annotations

import os
import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str | None
    default_model: str
    system_prompt: str
    max_tokens: int
    temperature: float


def _load_config() -> LLMConfig:
    api_key = (
        os.getenv("FOMC_REPORT_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set FOMC_REPORT_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY."
        )

    return LLMConfig(
        api_key=api_key,
        base_url=os.getenv("FOMC_REPORT_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        default_model=os.getenv("FOMC_REPORT_MODEL", "deepseek-reasoner"),
        system_prompt=os.getenv("FOMC_REPORT_SYSTEM_PROMPT", "You are a financial research assistant."),
        max_tokens=int(os.getenv("FOMC_REPORT_MAX_TOKENS", "4096")),
        temperature=float(os.getenv("FOMC_REPORT_TEMPERATURE", "0.6")),
    )


def _build_client(config: LLMConfig) -> OpenAI:
    kwargs: dict[str, str] = {"api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)


def chat_complete(prompt: str, model_name: str | None = None) -> tuple[str, str]:
    config = _load_config()
    client = _build_client(config)

    completion = client.chat.completions.create(
        model=model_name or config.default_model,
        messages=[
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stream=False,
    )

    response = completion.choices[0].message.content or ""
    reasoning = getattr(completion.choices[0].message, "reasoning_content", "") or ""
    return response, reasoning


def get_response(prompt: str, model_name: str | None = None, max_retry: int = 3) -> tuple[str, str]:
    last_response = ""
    last_reasoning = ""

    for retry in range(max_retry + 1):
        last_response, last_reasoning = chat_complete(prompt, model_name=model_name)
        if last_response.strip():
            return last_response, last_reasoning
        if retry < max_retry:
            time.sleep(5)

    return last_response, last_reasoning
