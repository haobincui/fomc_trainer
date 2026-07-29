from __future__ import annotations

import re
from dataclasses import dataclass


LEGACY_XML_TEMPLATE = "legacy_xml"
GEMMA4_THOUGHT_CHANNEL_TEMPLATE = "gemma4_thought_channel"
SUPPORTED_RESPONSE_TEMPLATES = (
    LEGACY_XML_TEMPLATE,
    GEMMA4_THOUGHT_CHANNEL_TEMPLATE,
)

_LEGACY_XML_PATTERN = re.compile(
    r"^\s*<think>(?P<reasoning>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)
_GEMMA4_THOUGHT_PATTERN = re.compile(
    r"^\s*<\|channel\>thought\n(?P<reasoning>.*?)(?:\n)?<channel\|>(?P<answer>.*)\s*$",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class ParsedResponse:
    format_name: str
    reasoning: str
    answer: str
    raw_text: str

    @property
    def has_explicit_reasoning(self) -> bool:
        return bool(self.reasoning.strip())


def normalize_response_template(value: str | None) -> str:
    template = str(value or LEGACY_XML_TEMPLATE).strip() or LEGACY_XML_TEMPLATE
    if template not in SUPPORTED_RESPONSE_TEMPLATES:
        raise ValueError(
            f"Unsupported response_template={template!r}. "
            f"Expected one of {SUPPORTED_RESPONSE_TEMPLATES}."
        )
    return template


def parse_response_text(text: str | None) -> ParsedResponse:
    raw_text = str(text or "").strip()
    if not raw_text:
        return ParsedResponse(format_name="plain", reasoning="", answer="", raw_text="")

    legacy_match = _LEGACY_XML_PATTERN.match(raw_text)
    if legacy_match:
        return ParsedResponse(
            format_name=LEGACY_XML_TEMPLATE,
            reasoning=legacy_match.group("reasoning").strip(),
            answer=legacy_match.group("answer").strip(),
            raw_text=raw_text,
        )

    gemma_match = _GEMMA4_THOUGHT_PATTERN.match(raw_text)
    if gemma_match:
        return ParsedResponse(
            format_name=GEMMA4_THOUGHT_CHANNEL_TEMPLATE,
            reasoning=gemma_match.group("reasoning").strip(),
            answer=gemma_match.group("answer").strip(),
            raw_text=raw_text,
        )

    return ParsedResponse(format_name="plain", reasoning="", answer=raw_text, raw_text=raw_text)


def render_response_text(
    reasoning: str | None,
    answer: str | None,
    *,
    response_template: str,
) -> str:
    template = normalize_response_template(response_template)
    reasoning_text = str(reasoning or "").strip()
    answer_text = str(answer or "").strip()

    if not reasoning_text:
        return answer_text

    if template == LEGACY_XML_TEMPLATE:
        return f"<think>{reasoning_text}</think><answer>{answer_text}</answer>"

    return f"<|channel>thought\n{reasoning_text}\n<channel|>{answer_text}"


def format_response_text(
    response_text: str | None,
    *,
    response_template: str,
    reasoning_text: str | None = None,
) -> str:
    parsed = parse_response_text(response_text)
    template = normalize_response_template(response_template)

    if reasoning_text is not None and str(reasoning_text).strip():
        explicit_reasoning = str(reasoning_text).strip()
        answer_text = parsed.answer if parsed.format_name != "plain" else str(response_text or "").strip()
        return render_response_text(
            explicit_reasoning,
            answer_text,
            response_template=template,
        )

    if parsed.format_name == template:
        return parsed.raw_text

    return render_response_text(
        parsed.reasoning,
        parsed.answer,
        response_template=template,
    )
