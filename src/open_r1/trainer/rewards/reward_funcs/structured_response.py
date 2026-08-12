"""Backward-compatible import path for the lightweight shared parser."""

from open_r1.structured_response import (
    DEEPSEEK_THINK_COMPLETION_FORMAT,
    GEMINI_THOUGHT_CHANNEL_FORMAT,
    LEGACY_XML_FORMAT,
    MIXED_FORMAT,
    PLAIN_TEXT_FORMAT,
    ParsedStructuredResponse,
    contains_deepseek_markers,
    contains_gemini_markers,
    contains_legacy_markers,
    extract_answer,
    extract_reasoning_and_answer,
    parse_structured_response,
)

__all__ = [
    "DEEPSEEK_THINK_COMPLETION_FORMAT",
    "GEMINI_THOUGHT_CHANNEL_FORMAT",
    "LEGACY_XML_FORMAT",
    "MIXED_FORMAT",
    "PLAIN_TEXT_FORMAT",
    "ParsedStructuredResponse",
    "contains_deepseek_markers",
    "contains_gemini_markers",
    "contains_legacy_markers",
    "extract_answer",
    "extract_reasoning_and_answer",
    "parse_structured_response",
]
