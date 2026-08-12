"""Frozen schemas and prompt contracts for chk1 data generation.

The strings in this module are deliberately free of same-meeting Minutes text,
policy decisions, and legacy targets.  Their hashes are bound into every
release so a prompt-contract change invalidates only the affected cache rows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


FACT_CARD_SCHEMA_VERSION = "chk1-point-in-time-fact-card-v1"
CANDIDATE_SCHEMA_VERSION = "chk1-deepseek-teacher-candidate-v1"
CRITIC_SCHEMA_VERSION = "chk1-local-critic-v1"
MANIFEST_SCHEMA_VERSION = "chk1-sft-manifest-row-v4"
RELEASE_SCHEMA_VERSION = "chk1-local-data-release-v1"
HANDOFF_SCHEMA_VERSION = "chk1-local-data-handoff-v1"
QUALITY_SCHEMA_VERSION = "chk1-local-data-quality-report-v1"
INVENTORY_SCHEMA_VERSION = "chk1-legacy-inventory-v1"

GENERATOR_SYSTEM_PROMPT = """\
You are the DeepSeek reasoning teacher for an economic-analysis SFT dataset.
Reason carefully and produce one concise FOMC-style briefing analysis using
only the supplied point-in-time fact card.

Hard rules:
1. Do not use outside knowledge, remembered events, people, policy decisions,
   same-meeting Minutes, or facts that are absent from the fact card.
2. Every number and every directional statement must be supported by one or
   more supplied evidence IDs. Do not invent precision, but do not cite or
   print evidence IDs inside answer; return them only in evidence_ids.
3. Distinguish observation from deterministic calculation. Do not assert a
   cause unless that cause is itself explicit evidence.
4. Express uncertainty in the supplied section style, without recommending or
   predicting a policy action.
5. Your provider reasoning_content is the distilled analysis. Return exactly
   one JSON object in content with keys answer and evidence_ids and no other
   keys or wrapper. answer must be non-empty plain prose: no JSON/schema text,
   evidence IDs, Markdown, markup, or model-control tokens. evidence_ids must
   be a separate, non-empty, duplicate-free string list. Do not copy the
   analysis into answer.
"""

CRITIC_SYSTEM_PROMPT = """\
You are an offline groundedness critic. Evaluate the candidate only against
the supplied point-in-time fact card and style guide. You must not use outside
knowledge or infer what happened at the meeting.

Return exactly one JSON object:
{"grounded": boolean, "unsupported_claims": [string], "style_score": integer,
 "reasoning_consistency": boolean}.

style_score is 1 through 5. Any unsupported number, event, person, policy
action, causal claim, or post-cutoff fact makes grounded false. Do not reward a
claim merely because it is historically true.
"""

REPAIR_SYSTEM_PROMPT = """\
Regenerate the rejected DeepSeek teacher target using only the supplied fact
card, style guide, and critic or output-contract error codes. Remove unsupported
content rather than replacing it with outside knowledge. Your new
reasoning_content is the replacement distilled analysis. Return exactly one
JSON object in content with keys answer and evidence_ids and no other text.
This is the only repair attempt.
"""


def canonical_json(value: Any) -> str:
    """Serialize finite JSON deterministically for hashing and cache keys."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_contract_hashes() -> dict[str, str]:
    return {
        "generator": sha256_text(GENERATOR_SYSTEM_PROMPT),
        "critic": sha256_text(CRITIC_SYSTEM_PROMPT),
        "repair": sha256_text(REPAIR_SYSTEM_PROMPT),
    }


def prompt_template_sha256() -> str:
    """Bind all three frozen local-generation prompt contracts with one digest."""

    return sha256_text(canonical_json(prompt_contract_hashes()))


def render_generator_payload(
    *,
    fact_card: Mapping[str, Any],
    atomic_topic: str,
    style_guide: Mapping[str, Any],
) -> str:
    """Render the only user payload that the local teacher may receive."""

    payload = {
        "atomic_topic": str(atomic_topic).strip(),
        "fact_card": dict(fact_card),
        "output_contract": {
            "type": "object",
            "provider_analysis_field": "reasoning_content",
            "required": ["answer", "evidence_ids"],
            "additionalProperties": False,
            "properties": {
                "answer": {"type": "string", "minLength": 1},
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                    "minItems": 1,
                },
            },
        },
        "section_style_guide": dict(style_guide),
    }
    rendered = canonical_json(payload)
    lowered = rendered.casefold()
    prohibited = (
        "reference_excerpt",
        "archived_response",
        "teacher_response",
        "rate_change",
        "current_rate",
        "decision_label",
    )
    found = [field for field in prohibited if f'"{field.casefold()}"' in lowered]
    if found:
        raise ValueError(f"Generator payload contains prohibited fields: {found}")
    return rendered


def render_student_prompt(
    *,
    fact_card: Mapping[str, Any],
    atomic_topic: str,
    style_guide: Mapping[str, Any],
) -> str:
    """Render the reference-free prompt stored in the canonical SFT split."""

    payload = render_generator_payload(
        fact_card=fact_card,
        atomic_topic=atomic_topic,
        style_guide=style_guide,
    )
    # The output schema helps the teacher but is not part of the student's
    # natural task. Rebuild the safe object without that teacher-only field.
    parsed = json.loads(payload)
    parsed.pop("output_contract")
    return (
        "Analyze the supplied point-in-time evidence for the atomic topic below. "
        "Use the supplied FOMC section style, remain neutral, and do not add "
        "external events, people, policy actions, causes, or post-cutoff facts. "
        "Every number and directional statement must be grounded in the fact "
        "card. Think first, then provide a concise final analysis.\n\n"
        + canonical_json(parsed)
    )
