from __future__ import annotations

import re
from dataclasses import dataclass


LEGACY_XML_FORMAT = "legacy_xml"
GEMINI_THOUGHT_CHANNEL_FORMAT = "gemini4_thought_channel"
DEEPSEEK_THINK_COMPLETION_FORMAT = "deepseek_think_completion"
MIXED_FORMAT = "mixed"
PLAIN_TEXT_FORMAT = "plain"

_LEGACY_MARKERS = ("<think>", "</think>", "<answer>", "</answer>")
_GEMINI_MARKERS = ("<|channel>thought", "<channel|>")
_DEEPSEEK_MARKERS = ("</think>",)
_LEGACY_PATTERN = re.compile(
    r"<think>\s*(.*?)\s*</think>\s*<answer>\s*(.*?)\s*</answer>",
    flags=re.DOTALL | re.IGNORECASE,
)
_GEMINI_PATTERN = re.compile(
    r"<\|channel>thought\n(.*?)<channel\|>(.*)",
    flags=re.DOTALL,
)
_DEEPSEEK_COMPLETION_PATTERN = re.compile(
    r"(.*?)\s*</think>\s*(.*)",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class ParsedStructuredResponse:
    reasoning: str
    answer: str
    format_name: str
    is_well_formed: bool


def contains_legacy_markers(text: str) -> bool:
    return isinstance(text, str) and any(marker in text for marker in ("<think>", "<answer>", "</answer>"))


def contains_gemini_markers(text: str) -> bool:
    return isinstance(text, str) and any(marker in text for marker in _GEMINI_MARKERS)


def contains_deepseek_markers(text: str) -> bool:
    return isinstance(text, str) and any(marker in text for marker in _DEEPSEEK_MARKERS)


def parse_structured_response(text: str) -> ParsedStructuredResponse:
    if not isinstance(text, str):
        return ParsedStructuredResponse("", "", PLAIN_TEXT_FORMAT, False)

    stripped = text.strip()
    if not stripped:
        return ParsedStructuredResponse("", "", PLAIN_TEXT_FORMAT, False)

    has_legacy = contains_legacy_markers(stripped)
    has_gemini = contains_gemini_markers(stripped)
    has_deepseek = contains_deepseek_markers(stripped) and not has_legacy

    active_formats = int(has_legacy) + int(has_gemini) + int(has_deepseek)
    if active_formats > 1:
        return ParsedStructuredResponse("", "", MIXED_FORMAT, False)

    if has_legacy:
        match = _LEGACY_PATTERN.fullmatch(stripped)
        if not match:
            return ParsedStructuredResponse("", "", LEGACY_XML_FORMAT, False)

        reasoning = match.group(1).strip()
        answer = match.group(2).strip()
        return ParsedStructuredResponse(
            reasoning=reasoning,
            answer=answer,
            format_name=LEGACY_XML_FORMAT,
            is_well_formed=bool(answer),
        )

    if has_gemini:
        match = _GEMINI_PATTERN.fullmatch(stripped)
        if not match:
            return ParsedStructuredResponse("", "", GEMINI_THOUGHT_CHANNEL_FORMAT, False)

        reasoning = match.group(1).strip()
        answer = match.group(2).strip()
        return ParsedStructuredResponse(
            reasoning=reasoning,
            answer=answer,
            format_name=GEMINI_THOUGHT_CHANNEL_FORMAT,
            is_well_formed=bool(answer),
        )

    if has_deepseek:
        match = _DEEPSEEK_COMPLETION_PATTERN.fullmatch(stripped)
        if not match:
            return ParsedStructuredResponse("", "", DEEPSEEK_THINK_COMPLETION_FORMAT, False)

        reasoning = match.group(1).strip()
        answer = match.group(2).strip()
        return ParsedStructuredResponse(
            reasoning=reasoning,
            answer=answer,
            format_name=DEEPSEEK_THINK_COMPLETION_FORMAT,
            is_well_formed=bool(answer),
        )

    return ParsedStructuredResponse("", stripped, PLAIN_TEXT_FORMAT, False)


def extract_reasoning_and_answer(
    text: str,
    *,
    allow_plain_answer_fallback: bool = True,
) -> tuple[str, str]:
    parsed = parse_structured_response(text)
    if parsed.is_well_formed:
        return parsed.reasoning, parsed.answer
    if parsed.format_name == PLAIN_TEXT_FORMAT and allow_plain_answer_fallback:
        return "", parsed.answer
    return "", ""


def extract_answer(
    text: str,
    *,
    allow_plain_answer_fallback: bool = False,
) -> str:
    parsed = parse_structured_response(text)
    if parsed.is_well_formed:
        return parsed.answer
    if parsed.format_name == PLAIN_TEXT_FORMAT and allow_plain_answer_fallback:
        return parsed.answer
    return ""
