from __future__ import annotations

import re


SECTION_CANONICAL_MAP = {
    "Participants' Views on Current Economic Conditions and the Economic Outlook": "Participants' Views on Current Conditions and the Economic Outlook",
    "â": "–",
    "â\x80\x93": "–",
}

ABNORMAL_TOPICS = {"Non-Core, Delete", "Delete", "Non-Core", "Other", ""}
DATE_LIKE_SECTION = re.compile(r"^[A-Za-z]+(?: [A-Za-z]+)? \d{1,2}(?:[–-]\d{1,2})?, \d{4}$")


def fix_mojibake(text: str) -> str:
    normalized = str(text or "").strip()
    for source, target in SECTION_CANONICAL_MAP.items():
        normalized = normalized.replace(source, target)
    normalized = normalized.replace("\xa0", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def canonicalize_section_name(section_name: str) -> str:
    normalized = fix_mojibake(section_name)
    return SECTION_CANONICAL_MAP.get(normalized, normalized)


def canonicalize_topic(topic: str) -> str:
    normalized = fix_mojibake(topic)
    return normalized.replace(" ,", ",")


def is_abnormal_section(section_name: str) -> bool:
    lowered = canonicalize_section_name(section_name).lower()
    return (
        bool(DATE_LIKE_SECTION.match(canonicalize_section_name(section_name)))
        or "videoconference meeting" in lowered
        or lowered == "minutes of the federal open market committee"
    )


def is_abnormal_topic(topic: str) -> bool:
    normalized = canonicalize_topic(topic)
    return normalized in ABNORMAL_TOPICS


def _is_divider_line(line: str) -> bool:
    return bool(re.fullmatch(r"\|\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|", line.strip()))


def table_metrics(markdown: str) -> dict[str, int | bool]:
    lines = [line.rstrip() for line in str(markdown or "").splitlines()]
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip().startswith("|"):
            current.append(line.strip())
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    data_tables = 0
    header_only_tables = 0
    for block in blocks:
        if len(block) < 2:
            header_only_tables += 1
            continue
        data_rows = [
            line
            for index, line in enumerate(block)
            if index > 1 and not _is_divider_line(line)
        ]
        if data_rows:
            data_tables += 1
        else:
            header_only_tables += 1

    return {
        "table_count": len(blocks),
        "data_table_count": data_tables,
        "header_only_table_count": header_only_tables,
        "has_nonempty_table": data_tables > 0,
    }


def prompt_length_metrics(prompt: str) -> dict[str, int]:
    return {
        "prompt_length_chars": len(prompt or ""),
        "prompt_length_words": len((prompt or "").split()),
    }
