from __future__ import annotations

from jobs.retrain_v2.chk1.topic_styles import (
    ATOMIC_TOPICS,
    canonical_atomic_topic,
    ledger_indicator_for_topic,
    topic_for_ledger_indicator,
    topic_style_map,
)


def test_frozen_roster_has_26_unique_topics_and_one_style_each() -> None:
    assert len(ATOMIC_TOPICS) == 26
    assert len(set(ATOMIC_TOPICS)) == 26
    mapping = topic_style_map()
    assert set(mapping) == set(ATOMIC_TOPICS)
    assert len(set(mapping.values())) == 2


def test_topic_and_ledger_names_round_trip() -> None:
    for topic in ATOMIC_TOPICS:
        indicator = ledger_indicator_for_topic(topic)
        assert topic_for_ledger_indicator(indicator) == topic
    assert canonical_atomic_topic("Consumer-Price-Index-(CPI)") == (
        "Consumer Price Index (CPI)"
    )
