from __future__ import annotations

import os
import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str | None
    model_name: str
    system_prompt: str
    max_tokens: int
    temperature: float
    max_retry: int
    sleep_seconds: float


def load_llm_config(payload: dict) -> LLMConfig:
    api_key_env = payload.get("api_key_env", "OPENAI_API_KEY")
    api_key = payload.get("api_key") or os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Provide payload.api_key or set env var: {api_key_env}"
        )
    base_url = payload.get("base_url") or os.getenv(payload.get("base_url_env", ""), "")
    return LLMConfig(
        api_key=api_key,
        base_url=base_url or None,
        model_name=payload["model_name"],
        system_prompt=payload.get("system_prompt", "You are a financial research assistant."),
        max_tokens=int(payload.get("max_tokens", 4096)),
        temperature=float(payload.get("temperature", 0.2)),
        max_retry=int(payload.get("max_retry", 3)),
        sleep_seconds=float(payload.get("sleep_seconds", 0.0)),
    )


def build_client(config: LLMConfig) -> OpenAI:
    kwargs: dict[str, str] = {"api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)


def get_response(prompt: str, config: LLMConfig) -> tuple[str, str]:
    client = build_client(config)
    last_response = ""
    last_reasoning = ""
    for retry in range(config.max_retry + 1):
        completion = client.chat.completions.create(
            model=config.model_name,
            messages=[
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stream=False,
        )
        last_response = completion.choices[0].message.content or ""
        last_reasoning = getattr(completion.choices[0].message, "reasoning_content", "") or ""
        if last_response.strip():
            if config.sleep_seconds:
                time.sleep(config.sleep_seconds)
            return last_response, last_reasoning
        if retry < config.max_retry:
            time.sleep(5)
    if config.sleep_seconds:
        time.sleep(config.sleep_seconds)
    return last_response, last_reasoning
