from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jobs.generation.generate_chk4_sft_targets import (
    TeacherResponse,
    canonical_json,
    sha256_file,
)
from jobs.generation.prepare_chk4_supplement import (
    ADMISSION_PROFILE,
    MIN_VALID_ATOMIC_TOPICS,
)
from jobs.generation.run_chk4_supplement_blind_v2 import (
    PROMPT_CONTRACT_FILE,
    run_blind_predictions_v2,
)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(max(1, len(text.split()))))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(message["content"] for message in messages) + "\nassistant:"


class QueueBackend:
    def __init__(self, contents: list[str]):
        self.contents = contents
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        config: Any,
        system_prompt: str,
        user_prompt: str,
        environment: Mapping[str, str] | None,
    ) -> TeacherResponse:
        del environment
        index = len(self.calls)
        self.calls.append(
            {
                "max_tokens": config.max_tokens,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return TeacherResponse(
            reasoning="provider provenance",
            content=self.contents[index],
            response_id=f"response-{index}",
            returned_model="deepseek-v4-pro",
            system_fingerprint="fp-fixed",
            finish_reason="stop",
            created=1,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_inputs(root: Path) -> None:
    sample_id = "dec-000000000000000000000000"
    admitted_path = root / "manifests/admitted.jsonl"
    _write_jsonl(
        admitted_path,
        [
            {
                "sample_id": sample_id,
                "meeting_date": "1993-03-23",
                "source_ids": ["source-0"],
                "admission_profile": ADMISSION_PROFILE,
                "gold": {"direction": "hold", "magnitude_bp": 0},
            }
        ],
    )
    _write_jsonl(
        root / "summaries/meeting_decision_briefs.jsonl",
        [
            {
                "sample_id": sample_id,
                "meeting_decision_brief": (
                    "Inflation pressures eased while employment activity and "
                    "financial conditions remained stable."
                ),
            }
        ],
    )
    handoff_payload_sha256 = "b" * 64
    handoff_path = root / "sources/materialized/source_handoff.json"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(
        canonical_json(
            {
                "schema_version": "chk1-source-handoff-v1",
                "payload_sha256": handoff_payload_sha256,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit_path = root / "reports/evidence_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        canonical_json(
            {
                "schema_version": "chk4-decision-supplement-evidence-audit-v2",
                "status": "complete",
                "admission_profile": ADMISSION_PROFILE,
                "candidate_count": 1,
                "admitted_count": 1,
                "rejected_count": 0,
                "minimum_admitted": 1,
                "minimum_valid_atomic_topics": MIN_VALID_ATOMIC_TOPICS,
                "source_handoff_payload_sha256": handoff_payload_sha256,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_blind_v2_uses_two_repairs_and_forbids_numeric_reasoning(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    backend = QueueBackend(
        [
            json.dumps(
                {
                    "reasoning": "A move of 25 basis points is appropriate.",
                    "direction": "hold",
                    "magnitude_bp": 0,
                }
            ),
            json.dumps(
                {
                    "reasoning": "Two considerations support maintaining the stance.",
                    "direction": "hold",
                    "magnitude_bp": 0,
                }
            ),
            json.dumps(
                {
                    "reasoning": (
                        "Easing inflation and steady activity support maintaining "
                        "the current stance."
                    ),
                    "direction": "hold",
                    "magnitude_bp": 0,
                }
            ),
        ]
    )

    summary = run_blind_predictions_v2(
        output_root=tmp_path,
        tokenizer_path=Path("unused"),
        concurrency=1,
        resume=False,
        backend=backend,
        tokenizer=FakeTokenizer(),
    )

    assert summary["status"] == "complete"
    assert summary["accepted_count"] == 1
    assert summary["reasoning_contract"] == "qualitative_no_numeric_quantities_v2"
    wrapper = Path(summary["execution_wrapper"]["path"])
    assert summary["execution_wrapper"]["sha256"] == sha256_file(wrapper)
    contract = json.loads(
        (tmp_path / "blind_predictions" / PROMPT_CONTRACT_FILE).read_text()
    )
    assert contract["repair_attempts"] == 2
    assert len(backend.calls) == 3
    prediction = json.loads(
        (tmp_path / "blind_predictions/predictions.jsonl").read_text().strip()
    )
    assert prediction["reasoning"].startswith("Easing inflation")
    assert all(
        "Canonical decision" not in call["user_prompt"] for call in backend.calls
    )
