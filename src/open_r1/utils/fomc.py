import json
import re
from functools import lru_cache
from pathlib import Path


def parse_boxed_vote(text: str) -> str:
    if not isinstance(text, str):
        return ""

    content = text.split("</think>")[-1]
    match = re.search(r"\\boxed\{(.*?)\}", content, re.DOTALL)
    return match.group(1).strip() if match else ""


def normalize_meeting_date(value) -> str:
    if value is None:
        return ""

    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")

    return str(value).strip()


@lru_cache(maxsize=4)
def load_rate_change_map(path: str = "dataset/processed/input_sources/rate_change_map.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
