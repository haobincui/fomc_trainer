"""Frozen atomic-topic roster and one-to-one section-style assignment."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .style_guide import stable_section_style_id


ECONOMIC_SECTION = "Staff Review of the Economic Situation"
FINANCIAL_SECTION = "Staff Review of the Financial Situation"

ECONOMIC_TOPICS = (
    "Business Investment",
    "Commodity Prices",
    "Consumer Confidence Index",
    "Consumer Price Index (CPI)",
    "GDP Growth",
    "Government Purchases",
    "Home Prices",
    "Housing Starts",
    "Industrial Production",
    "Labour Market",
    "Personal Consumption Expenditures (PCE)",
    "Trade Balance",
    "Unemployment Rate",
)

FINANCIAL_TOPICS = (
    "Bank Capital",
    "Bank Credit to Private Sector",
    "Corporate Bond Yields",
    "Equity Market Indices",
    "Exchange Rate",
    "Federal Funds Rate",
    "Federal Reserve Balance Sheet",
    "International Equity Markets",
    "Market Volatility (VIX)",
    "Money Supply",
    "Mortgage Rates",
    "Overnight Rate",
    "Treasury Yields",
)

ATOMIC_TOPICS = tuple(sorted((*ECONOMIC_TOPICS, *FINANCIAL_TOPICS)))

TOPIC_SECTION_NAMES = {
    **{topic: ECONOMIC_SECTION for topic in ECONOMIC_TOPICS},
    **{topic: FINANCIAL_SECTION for topic in FINANCIAL_TOPICS},
}


def topic_key(value: object) -> str:
    text = str(value or "").strip().casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", text)


_TOPIC_BY_KEY = {topic_key(topic): topic for topic in ATOMIC_TOPICS}
if len(_TOPIC_BY_KEY) != 26:  # defensive import-time freeze
    raise AssertionError("chk1 atomic-topic roster must contain exactly 26 topics")


def canonical_atomic_topic(value: object) -> str:
    key = topic_key(value)
    try:
        return _TOPIC_BY_KEY[key]
    except KeyError as exc:
        raise ValueError(f"Unknown chk1 atomic topic: {value!r}") from exc


def ledger_indicator_for_topic(value: object) -> str:
    """Map the display topic to the frozen LOO registry indicator slug."""

    topic = canonical_atomic_topic(value)
    return re.sub(r"[^A-Za-z0-9()]+", "-", topic).strip("-")


def topic_for_ledger_indicator(value: object) -> str:
    return canonical_atomic_topic(str(value).replace("-", " "))


def topic_style_map() -> dict[str, str]:
    return {
        topic: stable_section_style_id(TOPIC_SECTION_NAMES[topic])
        for topic in ATOMIC_TOPICS
    }


def style_entries_by_id(artifact: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    styles = artifact.get("styles")
    if not isinstance(styles, list):
        raise ValueError("style guide artifact has no styles list")
    by_id = {
        str(entry.get("section_style_id")): dict(entry)
        for entry in styles
        if isinstance(entry, Mapping) and entry.get("section_style_id")
    }
    missing = sorted(set(topic_style_map().values()) - set(by_id))
    if missing:
        raise ValueError(
            "Train Minutes did not produce all frozen topic style families: "
            f"{missing}"
        )
    return by_id


__all__ = [
    "ATOMIC_TOPICS",
    "ECONOMIC_SECTION",
    "FINANCIAL_SECTION",
    "TOPIC_SECTION_NAMES",
    "canonical_atomic_topic",
    "ledger_indicator_for_topic",
    "style_entries_by_id",
    "topic_for_ledger_indicator",
    "topic_key",
    "topic_style_map",
]
