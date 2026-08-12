from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jobs.generation.generate_chk4_sft_targets import (
    TeacherResponse,
    canonical_json,
    sha256_file,
)
from jobs.generation.prepare_chk4_supplement import (
    ADMISSION_PROFILE,
    MIN_VALID_ATOMIC_TOPICS,
    POPULATION,
)
from jobs.generation.run_chk4_supplement_teacher_v2 import (
    PROMPT_CONTRACT_FILE,
    build_parser,
    run_teacher_targets_v2,
)


SAMPLE_ID = "dec-000000000000000000000000"
BRIEF = (
    "Inflation eased while employment activity and financial conditions "
    "remained stable."
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
            reasoning="native provenance",
            content=self.contents[index],
            response_id=f"resp-{index}",
            returned_model="deepseek-v4-pro",
            system_fingerprint="fp-fixed",
            finish_reason="stop",
            created=1,
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _prepare_inputs(root: Path) -> None:
    handoff_payload_sha256 = "b" * 64
    _write_json(
        root / "sources/materialized/source_handoff.json",
        {
            "schema_version": "chk1-source-handoff-v1",
            "payload_sha256": handoff_payload_sha256,
        },
    )
    _write_json(
        root / "reports/evidence_audit.json",
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
        },
    )
    _write_jsonl(
        root / "manifests/admitted.jsonl",
        [
            {
                "sample_id": SAMPLE_ID,
                "meeting_date": "1993-01-01",
                "source_ids": ["source-0"],
                "admission_profile": ADMISSION_PROFILE,
                "gold": {"direction": "hold", "magnitude_bp": 0},
            }
        ],
    )
    _write_jsonl(
        root / "summaries/meeting_decision_briefs.jsonl",
        [{"sample_id": SAMPLE_ID, "meeting_decision_brief": BRIEF}],
    )


def _teacher_payload(reasoning: str) -> str:
    return json.dumps({"reasoning": reasoning, "direction": "hold", "magnitude_bp": 0})


def test_teacher_v2_uses_second_repair_and_preserves_output_families(
    tmp_path: Path,
) -> None:
    _prepare_inputs(tmp_path)
    backend = QueueBackend(
        [
            _teacher_payload("One inflation signal supports maintaining the stance."),
            _teacher_payload("Two activity signals support maintaining the stance."),
            _teacher_payload(
                "Easing inflation and stable activity support maintaining the stance."
            ),
        ]
    )

    result = run_teacher_targets_v2(
        output_root=tmp_path,
        tokenizer_path=Path("unused"),
        concurrency=1,
        resume=False,
        backend=backend,
        tokenizer=FakeTokenizer(),
    )

    stage_root = tmp_path / "teacher_targets/supplement"
    contract = json.loads((stage_root / PROMPT_CONTRACT_FILE).read_text())
    assert contract["repair_attempts"] == 2
    assert result["accepted_count"] == 1
    assert result["failure_count"] == 0
    assert result["api_requests"] == 1
    assert len(backend.calls) == 3

    expected_files = {
        stage_root / family / f"{split}.jsonl"
        for family in ("prepared", "manifests", "sft", "teacher_responses")
        for split in ("train", "validation", "test")
    }
    assert all(path.is_file() for path in expected_files)
    assert all(
        path.read_bytes() == b""
        for path in expected_files
        if path.name in {"validation.jsonl", "test.jsonl"}
    )

    prepared = _read_jsonl(stage_root / "prepared/train.jsonl")
    manifests = _read_jsonl(stage_root / "manifests/train.jsonl")
    sft = _read_jsonl(stage_root / "sft/train.jsonl")
    responses = _read_jsonl(stage_root / "teacher_responses/train.jsonl")
    assert len(prepared) == len(manifests) == len(sft) == len(responses) == 1
    assert manifests[0]["population"] == POPULATION
    assert manifests[0]["population_role"] == "supplement"
    assert manifests[0]["contract_sha256"] == contract["contract_sha256"]
    assert responses[0]["attempt"] == "repair_2"
    assert sft[0]["response"].endswith(
        '</think>\n{"direction":"hold","magnitude_bp":0}'
    )

    wrapper = Path(result["execution_wrapper"]["path"])
    assert wrapper.name == "run_chk4_supplement_teacher_v2.py"
    assert result["execution_wrapper"]["sha256"] == sha256_file(wrapper)
    saved_summary = json.loads((stage_root / "summary.json").read_text())
    assert canonical_json(saved_summary) == canonical_json(result)


def test_teacher_v2_cli_exposes_scoped_runtime_flags(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path / "release"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--concurrency",
            "3",
            "--resume",
        ]
    )
    assert args.output_root == tmp_path / "release"
    assert args.tokenizer == tmp_path / "tokenizer"
    assert args.concurrency == 3
    assert args.resume is True
