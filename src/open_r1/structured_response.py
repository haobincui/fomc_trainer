"""Lightweight structured-response parsing shared by rewards and generation."""

from __future__ import annotations

import re
from dataclasses import dataclass


LEGACY_XML_FORMAT = "legacy_xml"
GEMINI_THOUGHT_CHANNEL_FORMAT = "gemini4_thought_channel"
DEEPSEEK_THINK_COMPLETION_FORMAT = "deepseek_think_completion"
MIXED_FORMAT = "mixed"
PLAIN_TEXT_FORMAT = "plain"

LENGTH_TOLERANT_ANSWER_TAG_MODE = "answer_tag"
LENGTH_TOLERANT_FULL_COMPLETION_MODE = "full_completion"
LENGTH_TOLERANT_EMPTY_MODE = "empty"

_GEMINI_MARKERS = ("<|channel>thought", "<channel|>")
_DEEPSEEK_MARKERS = ("</think>",)
_LEADING_THINK_OPEN_PATTERN = re.compile(r"\A\s*<think>", flags=re.IGNORECASE)
_ANSWER_OPEN_PATTERN = re.compile(r"<answer>", flags=re.IGNORECASE)
_TRAILING_ANSWER_CLOSE_PATTERN = re.compile(
    r"\s*</answer>\s*\Z",
    flags=re.IGNORECASE,
)
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


@dataclass(frozen=True)
class LengthTolerantCandidate:
    """Candidate text and marker audit data for length-truncated completions."""

    text: str
    extraction_mode: str
    has_leading_think_tag: bool
    has_answer_opening_tag: bool
    has_trailing_answer_closing_tag: bool

    @property
    def is_nonempty(self) -> bool:
        return bool(self.text)


def extract_length_tolerant_candidate(text: str) -> LengthTolerantCandidate:
    """Extract scoreable text without requiring a normally terminated response.

    This intentionally does not call or modify the strict structured-response
    parser. The first case-insensitive ``<answer>`` opening tag is authoritative
    when present. Otherwise, the complete non-empty completion is the candidate.
    """

    if not isinstance(text, str):
        return LengthTolerantCandidate(
            text="",
            extraction_mode=LENGTH_TOLERANT_EMPTY_MODE,
            has_leading_think_tag=False,
            has_answer_opening_tag=False,
            has_trailing_answer_closing_tag=False,
        )

    stripped = text.strip()
    if not stripped:
        return LengthTolerantCandidate(
            text="",
            extraction_mode=LENGTH_TOLERANT_EMPTY_MODE,
            has_leading_think_tag=False,
            has_answer_opening_tag=False,
            has_trailing_answer_closing_tag=False,
        )

    has_leading_think_tag = bool(_LEADING_THINK_OPEN_PATTERN.search(stripped))
    answer_open = _ANSWER_OPEN_PATTERN.search(stripped)
    if answer_open is None:
        return LengthTolerantCandidate(
            text=stripped,
            extraction_mode=LENGTH_TOLERANT_FULL_COMPLETION_MODE,
            has_leading_think_tag=has_leading_think_tag,
            has_answer_opening_tag=False,
            has_trailing_answer_closing_tag=False,
        )

    candidate = stripped[answer_open.end() :]
    trailing_close = _TRAILING_ANSWER_CLOSE_PATTERN.search(candidate)
    if trailing_close is not None:
        candidate = candidate[: trailing_close.start()]

    return LengthTolerantCandidate(
        text=candidate.strip(),
        extraction_mode=LENGTH_TOLERANT_ANSWER_TAG_MODE,
        has_leading_think_tag=has_leading_think_tag,
        has_answer_opening_tag=True,
        has_trailing_answer_closing_tag=trailing_close is not None,
    )


def contains_legacy_markers(text: str) -> bool:
    return isinstance(text, str) and any(
        marker in text for marker in ("<think>", "<answer>", "</answer>")
    )


def contains_gemini_markers(text: str) -> bool:
    return isinstance(text, str) and any(marker in text for marker in _GEMINI_MARKERS)


def contains_deepseek_markers(text: str) -> bool:
    return isinstance(text, str) and any(
        marker in text for marker in _DEEPSEEK_MARKERS
    )


def parse_structured_response(text: str) -> ParsedStructuredResponse:
    if not isinstance(text, str):
        return ParsedStructuredResponse("", "", PLAIN_TEXT_FORMAT, False)

    stripped = text.strip()
    if not stripped:
        return ParsedStructuredResponse("", "", PLAIN_TEXT_FORMAT, False)

    has_legacy = contains_legacy_markers(stripped)
    has_gemini = contains_gemini_markers(stripped)
    has_deepseek_boundary = contains_deepseek_markers(stripped)

    # Gemini control tokens cannot be combined safely with either XML or the
    # native DeepSeek boundary.  DeepSeek's optional opening <think> tag is not
    # considered a legacy-format conflict: the first </think> is authoritative.
    if has_gemini and (has_legacy or has_deepseek_boundary):
        return ParsedStructuredResponse("", "", MIXED_FORMAT, False)

    legacy_match = _LEGACY_PATTERN.fullmatch(stripped)
    if legacy_match:
        reasoning = legacy_match.group(1).strip()
        answer = legacy_match.group(2).strip()
        return ParsedStructuredResponse(
            reasoning=reasoning,
            answer=answer,
            format_name=LEGACY_XML_FORMAT,
            is_well_formed=bool(answer),
        )

    if has_gemini:
        match = _GEMINI_PATTERN.fullmatch(stripped)
        if not match:
            return ParsedStructuredResponse(
                "", "", GEMINI_THOUGHT_CHANNEL_FORMAT, False
            )
        reasoning = match.group(1).strip()
        answer = match.group(2).strip()
        return ParsedStructuredResponse(
            reasoning=reasoning,
            answer=answer,
            format_name=GEMINI_THOUGHT_CHANNEL_FORMAT,
            is_well_formed=bool(answer),
        )

    if has_deepseek_boundary:
        # The first native closing boundary is the only required separator.
        # Everything after it is final-answer text; no <answer> wrapper is
        # required.  Some SFT targets emit an explicit opening <think>, while
        # the stock DeepSeek chat template supplies it outside decoded tokens.
        match = _DEEPSEEK_COMPLETION_PATTERN.fullmatch(stripped)
        if not match:
            return ParsedStructuredResponse(
                "", "", DEEPSEEK_THINK_COMPLETION_FORMAT, False
            )
        reasoning = match.group(1).strip()
        reasoning = _LEADING_THINK_OPEN_PATTERN.sub("", reasoning, count=1).strip()
        answer = match.group(2).strip()
        return ParsedStructuredResponse(
            reasoning=reasoning,
            answer=answer,
            format_name=DEEPSEEK_THINK_COMPLETION_FORMAT,
            is_well_formed=bool(answer),
        )

    if has_legacy:
        return ParsedStructuredResponse("", "", LEGACY_XML_FORMAT, False)

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
