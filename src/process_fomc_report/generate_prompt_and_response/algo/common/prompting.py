from __future__ import annotations

import re
from pathlib import Path

from .io_utils import sha256_text
from .quality import canonicalize_section_name, canonicalize_topic


MEETING_PATTERNS = [
    re.compile(r"meeting on \*\*(\d{4}-\d{2}-\d{2})(?: \d{2}:\d{2}:\d{2})?\*\*", re.IGNORECASE),
    re.compile(r"for the \*\*(\d{4}-\d{2}-\d{2})\*\* meeting", re.IGNORECASE),
    re.compile(r"upcoming meeting on \*\*(\d{4}-\d{2}-\d{2})\*\*", re.IGNORECASE),
]

SECTION_PATTERNS = [
    re.compile(r"tone and style of the \*\*(.*?)\*\* section", re.IGNORECASE | re.DOTALL),
    re.compile(r"aligned with the tone and structure of the \*\*(.*?)\*\* section", re.IGNORECASE | re.DOTALL),
]

TOPIC_PATTERNS = [
    re.compile(r"recent trends in \*\*(.*?)\*\* and related indicators", re.IGNORECASE | re.DOTALL),
]

REFERENCE_EXCERPT_PATTERNS = [
    re.compile(r"following excerpt from the FOMC minutes:\s*\n+\{\s*\n?(.*?)\n?\}\s*\n+\s*Keep your response", re.IGNORECASE | re.DOTALL),
    re.compile(r"following excerpt from the FOMC minutes:\s*\n+\{\s*\n?(.*?)\n?\}\s*$", re.IGNORECASE | re.DOTALL),
]

CURRENT_RATE_PATTERNS = [
    re.compile(r"stands at \*\*([0-9.]+)%\*\*", re.IGNORECASE),
    re.compile(r"is \*\*([0-9.]+)%\*\*", re.IGNORECASE),
]

TARGET_SECTION_PATTERNS = [
    re.compile(r"\*\*<Target Section>\*\*\s*(.*?)\s*\*\*</Target Section>\*\*", re.IGNORECASE | re.DOTALL),
    re.compile(r"aligned with the tone and structure of the \*\*(.*?)\*\* section", re.IGNORECASE | re.DOTALL),
]

POLICY_OPTIONS_PATTERNS = [
    re.compile(
        r"(?:Choose only one of the following:|Cast your vote.*?selecting exactly one option from the list below\.|Conclude with exactly one policy vote chosen from the options provided below\.)\s*(.*?)(?:\n\n(?:Be thoughtful|Guidelines:|---|## ))",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"### Allowed Policy Options\s*(.*?)(?:\n\n###|\Z)", re.IGNORECASE | re.DOTALL),
]

ANALYSIS_TEXT_PATTERNS = [
    re.compile(r"\*\*<Raw Analysis>\*\*\s*(.*?)\s*\*\*</Raw Analysis>\*\*", re.IGNORECASE | re.DOTALL),
    re.compile(r"## Staff Economic and Financial Market Analysis\s*(.*)\Z", re.IGNORECASE | re.DOTALL),
    re.compile(r"## Current Economic and Financial Market Analysis\s*(.*)\Z", re.IGNORECASE | re.DOTALL),
]


def load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_first(text: str, patterns: list[re.Pattern[str]]) -> str:
    if not isinstance(text, str):
        return ""
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return ""


def extract_meeting_date(prompt: str, fallback: str | None = None) -> str:
    if fallback:
        return str(fallback).strip()[:10]
    return extract_first(prompt, MEETING_PATTERNS)


def extract_section_name(prompt: str, fallback: str | None = None) -> str:
    if fallback:
        return canonicalize_section_name(fallback)
    return canonicalize_section_name(extract_first(prompt, SECTION_PATTERNS))


def extract_topic(prompt: str, fallback: str | None = None) -> str:
    if fallback:
        return canonicalize_topic(fallback)
    return canonicalize_topic(extract_first(prompt, TOPIC_PATTERNS))


def extract_reference_excerpt(prompt: str) -> str:
    return extract_first(prompt, REFERENCE_EXCERPT_PATTERNS)


def extract_current_rate(prompt: str, fallback=None) -> float | None:
    if fallback is not None and str(fallback) != "":
        return float(fallback)
    value = extract_first(prompt, CURRENT_RATE_PATTERNS)
    return float(value) if value else None


def _clean_block(text: str) -> str:
    return str(text or "").strip()


def extract_target_section(prompt: str, fallback: str | None = None) -> str:
    if fallback:
        return canonicalize_section_name(fallback)
    return canonicalize_section_name(extract_first(prompt, TARGET_SECTION_PATTERNS))


def extract_raw_analysis(prompt: str) -> str:
    matches = []
    for pattern in ANALYSIS_TEXT_PATTERNS:
        matches.extend(pattern.finditer(prompt or ""))
    if not matches:
        return ""
    return _clean_block(matches[-1].group(1))


def extract_policy_options(prompt: str) -> str:
    text = extract_first(prompt, POLICY_OPTIONS_PATTERNS)
    return _clean_block(text)


def render_analysis_prompt(
    *,
    template: str,
    meeting_date: str,
    topic: str,
    section_style: str,
    data_label: str,
    table_str: str,
    reference_excerpt: str = "",
) -> str:
    return template.format(
        meeting_date=meeting_date,
        topic=topic,
        section_style=section_style,
        emphasized_label=data_label,
        table_str=table_str.strip(),
        reference_excerpt=reference_excerpt.strip(),
    ).strip() + "\n"


def render_rewrite_prompt(
    *,
    template: str,
    meeting_date: str,
    target_section: str,
    raw_analysis: str,
) -> str:
    return template.format(
        meeting_date=meeting_date,
        target_section=target_section,
        raw_analysis=raw_analysis.strip(),
    ).strip() + "\n"


def render_decision_prompt(
    *,
    template: str,
    meeting_date: str,
    current_rate: float | str | None,
    analysis_text: str,
    policy_options: str,
) -> str:
    current_rate_text = "" if current_rate in (None, "") else str(current_rate)
    return template.format(
        meeting_date=meeting_date,
        current_rate=current_rate_text,
        analysis_text=analysis_text.strip(),
        policy_options=policy_options.strip(),
    ).strip() + "\n"


def prompt_hash(prompt: str) -> str:
    return sha256_text(prompt)
