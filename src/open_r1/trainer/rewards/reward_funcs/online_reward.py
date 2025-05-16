import json
import os
import re
import requests
import time
from pathlib import Path

# _URL = "http://localhost:8000/v1/chat/completions"
# _MODEL = "models/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit"
_API_KEY = None

_URL = "http://10.30.58.139:11432/api/chat/"
_MODEL = "gemma3:12b"




_ANSWER_PROMPT = (Path(__file__).parent / "online_reward_prompt/output_evaluation.md").resolve().read_text()
_REASONING_PROMPT = (Path(__file__).parent / "online_reward_prompt/reasoning_process_evaluation.md").resolve().read_text()
_COMBINED_PROMPT = (Path(__file__).parent / "online_reward_prompt/online_combined_evaluation.md").resolve().read_text()


# _SYSTEM_PROMPT = (
#     "Wrap the reasoning process in <think> and </think> tags, while the final answer should be enclosed within <answer> and </answer> tags. "
#     "The total score should be reported as: **Total Score**: \\boxed{{total_score}}"
# )

_SYSTEM_PROMPT = """
You are an instruction-following assistant that generates structured analytical outputs in two stages:

1. First, you must enclose your reasoning process within `<think>...</think>` tags.
   - This section should reflect your internal thought process, including any analysis, logic, or intermediate steps.

2. Then, provide your final conclusion or answer wrapped inside `<answer>...</answer>` tags.
   - This should be a concise summary, recommendation, or final judgment derived from your reasoning.

At the end of your response, you must include a total score using the **exact** format:

**Total Score**: \\boxed{XX}

Where `XX` is the integer score from 1 to 35.

---

Format Enforcement Rules:
- Do **not** include any explanation outside the `<think>` or `<answer>` blocks.
- Do **not** include headings, bullet points, or free text before or after the tags.
- Output must always end with the `**Total Score**: \boxed{XX}` line.
- Ensure only **one** `<think>` and **one** `<answer>` block per response.

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
    if "</think>" in text:
        think, answer = text.split("</think>", 1)
        return think.strip(), answer.strip()
    else:
        return "", text.strip()



def _send_eval_request(prompt: str, url: str) -> float:
    headers = {
        "Content-Type": "application/json",
    }
    if _API_KEY:
        headers["Authorization"] = f"Bearer {_API_KEY}"

    body = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "keep_alive": -1,
        "temperature": 0.3,
        "max_tokens": 2048,
        "top_p": 0.9,
    }

    try:
        resp = requests.post(url,headers=headers, json=body, timeout=180)
        resp.raise_for_status()
        result_text = resp.json().get("message", {}).get("content", "")
        # result_text = resp.json()["choices"][0]["message"]["content"]
        print(result_text)
        return _parse_score(result_text)
    except Exception as e:
        print(f"❌ Error during reward call: {e}")
        return 0.0

def _parse_answer(text: str):
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""


def answer_reward(
    completions: list[list[dict[str, str]]],
    response: list[str],
    provided_data: list[str],
    save_path: str = None,
    **kwargs
) -> list[float]:
    """Reward based on model answer quality."""
    rewards = []
    idx = 0
    for completion, reference, pdata in zip(completions, response, provided_data):
        content = completion[0]["content"]
        _, model_answer = _parse_reasoning_and_answer(content)
        prompt = _ANSWER_PROMPT.format(provided_data = pdata, reference_analysis=_parse_answer(reference), model_analysis=model_answer)
        score = _send_eval_request(prompt, _URL)
        rewards.append(score)
        idx += 1
        time.sleep(0.5)
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
    **kwargs
) -> list[float]:
    """Reward based on model reasoning quality."""
    rewards = []
    idx = 0
    for completion, reference in zip(completions, response):
        content = completion[0]["content"]
        model_reasoning, model_analysis = _parse_reasoning_and_answer(content)
        prompt = _REASONING_PROMPT.format(model_analysis=model_analysis, model_reasoning=model_reasoning)
        score = _send_eval_request(prompt, _URL)
        rewards.append(score)
        idx += 1
        time.sleep(0.5)
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
    **kwargs
) -> list[float]:
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
        score = _send_eval_request(prompt, _URL)
        rewards.append(score)
        idx += 1
        time.sleep(0.5)
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

