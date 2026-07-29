import json
import os
import re
import requests
import time
from pathlib import Path

from open_r1.trainer.rewards.reward_funcs.structured_response import (
    extract_answer,
    extract_reasoning_and_answer,
)

DEFAULT_JUDGE_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "models/gemma-3-12b-it"
DEFAULT_API_KEY_ENV = "OPEN_R1_JUDGE_API_KEY"




_ANSWER_PROMPT = (Path(__file__).parent / "online_reward_prompt/output_evaluation.md").resolve().read_text()
_REASONING_PROMPT = (Path(__file__).parent / "online_reward_prompt/reasoning_process_evaluation.md").resolve().read_text()
_COMBINED_PROMPT = (Path(__file__).parent / "online_reward_prompt/online_combined_evaluation.md").resolve().read_text()


# _SYSTEM_PROMPT = (
#     "Wrap the reasoning process in <think> and </think> tags, while the final answer should be enclosed within <answer> and </answer> tags. "
#     "The total score should be reported as: **Total Score**: \\boxed{{total_score}}"
# )

_SYSTEM_PROMPT = """
You are an instruction-following evaluator for economic policy analysis outputs.

You may receive model outputs that were originally produced in either of these response styles:

1. Legacy XML style with `<think>...</think>` and `<answer>...</answer>`
2. Gemini/Gemma thought-channel style with `<|channel>thought ... <channel|>final answer`

Your job is not to preserve or imitate the input response format. You only need to evaluate the supplied content and return a score plus concise justification.

At the end of your response, you must include a total score using the **exact** format:

**Total Score**: \\boxed{XX}

Where `XX` is the integer score from 1 to 35.

---

Format Enforcement Rules:
- Do **not** include XML tags, channel tags, or any other wrapper structure.
- Do **not** include headings or free text before the score and short justification.
- Output must always end with the `**Total Score**: \\boxed{{XX}}` line.

Follow these formatting instructions strictly.
"""


def save_judge_record(save_path, record):
    """Append a single record to the jsonl log file."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")




def _parse_score(text: str) -> float:
    pattern = r"\s*\\boxed\{(\d+)\}"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else 0



# def _parse_reasoning_and_answer(text: str) -> tuple[str, str]:
#     pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"
#     match = re.search(pattern, text, re.DOTALL)
#     return (match.group(1).strip(), match.group(2).strip()) if match else ("", "")

def _parse_reasoning_and_answer(text: str) -> tuple[str, str]:
    return extract_reasoning_and_answer(text, allow_plain_answer_fallback=True)


def get_online_reward_settings(
    url: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
    verbose: bool | None = None,
    sleep_seconds: float | None = None,
    api_key_env: str | None = DEFAULT_API_KEY_ENV,
) -> dict:
    resolved_api_key_env = api_key_env or DEFAULT_API_KEY_ENV
    env_verbose = os.environ.get("OPEN_R1_JUDGE_VERBOSE")
    env_timeout = os.environ.get("OPEN_R1_JUDGE_TIMEOUT")
    env_sleep = os.environ.get("OPEN_R1_JUDGE_SLEEP_SECONDS")
    env_api_key = os.environ.get(resolved_api_key_env) if resolved_api_key_env else None

    return {
        "url": url or os.environ.get("OPEN_R1_JUDGE_URL", DEFAULT_JUDGE_URL),
        "model": model or os.environ.get("OPEN_R1_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        "timeout": int(timeout if timeout is not None else env_timeout or 180),
        "verbose": bool(
            verbose if verbose is not None else env_verbose in {"1", "true", "TRUE", "yes", "YES"}
        ),
        "sleep_seconds": float(sleep_seconds if sleep_seconds is not None else env_sleep or 0.0),
        "api_key_env": resolved_api_key_env,
        "api_key": env_api_key,
    }



def _send_eval_request(
    prompt: str,
    *,
    url: str,
    model: str,
    timeout: int,
    api_key: str | None,
    verbose: bool,
    api_key_env: str | None = None,
    sleep_seconds: float | None = None,
) -> float:
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "temperature": 0.3,
        "max_tokens": 3072,
        "top_p": 0.9,
    }
    if not url.rstrip("/").endswith("/v1/chat/completions"):
        body["keep_alive"] = -1

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        result_text = (
            payload.get("message", {}).get("content", "")
            or payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        if verbose:
            print("\n=============\n")
            print(result_text)
            print("\n=============\n")
        return _parse_score(result_text)
    except Exception as e:
        if verbose:
            print(f"❌ Error during reward call: {e}")
        return 0.0

def _parse_answer(text: str):
    return extract_answer(text)


def answer_reward(
    completions: list[list[dict[str, str]]],
    response: list[str],
    provided_data: list[str],
    save_path: str = None,
    url: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
    verbose: bool | None = None,
    sleep_seconds: float | None = None,
    api_key_env: str | None = DEFAULT_API_KEY_ENV,
    **kwargs
) -> list[float]:
    """Reward based on model answer quality."""
    settings = get_online_reward_settings(
        url=url,
        model=model,
        timeout=timeout,
        verbose=verbose,
        sleep_seconds=sleep_seconds,
        api_key_env=api_key_env,
    )
    rewards = []
    idx = 0
    for completion, reference, pdata in zip(completions, response, provided_data):
        content = completion[0]["content"]
        _, model_answer = _parse_reasoning_and_answer(content)
        prompt = _ANSWER_PROMPT.format(provided_data = pdata, reference_analysis=_parse_answer(reference), model_analysis=model_answer)
        score = _send_eval_request(prompt, **settings) / 35
        rewards.append(score)
        idx += 1
        if settings["sleep_seconds"] > 0:
            time.sleep(settings["sleep_seconds"])
        if save_path:
            save_judge_record(save_path, {
                "type": "answer",
                "index": idx,
                "input": {
                    "provided_data": pdata,
                    "reference_analysis": reference,
                    "model_analysis": model_answer
                },
                "score": score
            })
    return rewards


def reasoning_reward(
    completions: list[list[dict[str, str]]],
    response: list[str],
    save_path: str = None,
    url: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
    verbose: bool | None = None,
    sleep_seconds: float | None = None,
    api_key_env: str | None = DEFAULT_API_KEY_ENV,
    **kwargs
) -> list[float]:
    """Reward based on model reasoning quality."""
    settings = get_online_reward_settings(
        url=url,
        model=model,
        timeout=timeout,
        verbose=verbose,
        sleep_seconds=sleep_seconds,
        api_key_env=api_key_env,
    )
    rewards = []
    idx = 0
    for completion, reference in zip(completions, response):
        content = completion[0]["content"]
        model_reasoning, model_analysis = _parse_reasoning_and_answer(content)
        prompt = _REASONING_PROMPT.format(model_analysis=model_analysis, model_reasoning=model_reasoning)
        score = _send_eval_request(prompt, **settings) / 5
        rewards.append(score)
        idx += 1
        if settings["sleep_seconds"] > 0:
            time.sleep(settings["sleep_seconds"])
        if save_path:
            save_judge_record(save_path, {
                "type": "reasoning",
                "index": idx,
                "input": {
                    "reference_analysis": reference,
                    "model_reasoning": model_reasoning,
                    "model_analysis": model_analysis
                },
                "score": score
            })
    return rewards


def combined_reward(
    completions: list[list[dict[str, str]]],
    response: list[str],
    provided_data: list[str],
    save_path: str = None,
    url: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
    verbose: bool | None = None,
    sleep_seconds: float | None = None,
    api_key_env: str | None = DEFAULT_API_KEY_ENV,
    **kwargs
) -> list[float]:
    settings = get_online_reward_settings(
        url=url,
        model=model,
        timeout=timeout,
        verbose=verbose,
        sleep_seconds=sleep_seconds,
        api_key_env=api_key_env,
    )
    rewards = []
    idx = 0
    for completion, reference, pdata in zip(completions, response, provided_data):
        content = completion[0]["content"]
        model_reasoning, model_analysis = _parse_reasoning_and_answer(content)
        reference_analysis = _parse_answer(reference)

        prompt = _COMBINED_PROMPT.format(
            provided_data = pdata,
            reference_analysis=reference_analysis,
            model_analysis=model_analysis,
            model_reasoning=model_reasoning)
        score = _send_eval_request(prompt, **settings)
        rewards.append(score)
        idx += 1
        if settings["sleep_seconds"] > 0:
            time.sleep(settings["sleep_seconds"])
        if save_path:
            save_judge_record(save_path, {
                "type": "combined",
                "index": idx,
                "input": {
                    "reference_analysis": reference,
                    "model_reasoning": model_reasoning,
                    "model_analysis": model_analysis,
                    "provided_data": pdata
                },
                "score": score
            })
    return rewards
