"""Run the versioned qualitative teacher-target stage for the chk4 supplement."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.generation.generate_chk4_sft_targets import (
    DEFAULT_TOKENIZER,
    MANIFEST_SCHEMA,
    render_student_prompt,
    sha256_file,
    sha256_text,
)
from jobs.generation.generate_chk4_supplement import (
    DECISION_MAX_TOKENS,
    DEFAULT_CONCURRENCY,
    DEFAULT_OUTPUT_ROOT,
    TARGET_REPAIR_SYSTEM_PROMPT,
    TARGET_SCHEMA,
    TARGET_SYSTEM_PROMPT,
    _canonical_gold,
    _decision_rows,
    _evidence_profile_binding,
    _load_tokenizer,
    _run_stage,
    _update_root_summary,
    _validate_target,
    _write_json,
    _write_jsonl,
)
from jobs.generation.prepare_chk4_supplement import POPULATION


PROMPT_CONTRACT_FILE = "prompt_contract.qualitative-v2.json"


def run_teacher_targets_v2(
    *,
    output_root: Path,
    tokenizer_path: Path,
    concurrency: int,
    resume: bool,
    backend: Any | None = None,
    environment: Mapping[str, str] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Generate the supplement teacher release under the v2 repair contract."""

    evidence_binding = _evidence_profile_binding(output_root)
    rows = _decision_rows(output_root, include_gold=True)
    tokenizer = tokenizer or _load_tokenizer(tokenizer_path)
    stage_root = output_root / "teacher_targets/supplement"
    accepted, summary = _run_stage(
        stage="teacher_targets",
        rows=rows,
        output_root=output_root,
        stage_root=stage_root,
        system_prompt=TARGET_SYSTEM_PROMPT,
        repair_prompt=TARGET_REPAIR_SYSTEM_PROMPT,
        max_tokens=DECISION_MAX_TOKENS,
        tokenizer=tokenizer,
        validator=_validate_target,
        concurrency=concurrency,
        resume=resume,
        backend=backend,
        environment=environment,
        prompt_contract_file=PROMPT_CONTRACT_FILE,
        repair_attempts=2,
    )

    prepared_rows = []
    manifest_rows = []
    sft_rows = []
    teacher_rows = []
    for row in rows:
        payload = accepted[row.sample_id]
        student_prompt = render_student_prompt(row.input_text)
        gold_text = _canonical_gold(row.direction, row.magnitude_bp)
        prepared_rows.append(
            {
                "schema_version": TARGET_SCHEMA,
                "sample_id": row.sample_id,
                "prompt": student_prompt,
                "input_sha256": row.input_sha256,
                "prompt_sha256": sha256_text(student_prompt),
                "gold_sha256": sha256_text(gold_text),
            }
        )
        manifest_rows.append(
            {
                "schema_version": MANIFEST_SCHEMA,
                "sample_id": row.sample_id,
                "meeting_date": row.meeting_date,
                "split": "train",
                "population": POPULATION,
                "population_role": "supplement",
                "admission_profile": evidence_binding["admission_profile"],
                "evidence_audit_sha256": evidence_binding["evidence_audit"]["sha256"],
                "source_handoff_payload_sha256": evidence_binding["source_handoff"][
                    "payload_sha256"
                ],
                "source_ids": list(row.source_ids),
                "gold": row.gold,
                "input_sha256": row.input_sha256,
                "prompt_sha256": sha256_text(student_prompt),
                "gold_sha256": sha256_text(gold_text),
                "contract_sha256": summary["contract_sha256"],
            }
        )
        sft_rows.append(
            {"prompt": student_prompt, "response": payload["target"]["completion"]}
        )
        teacher_rows.append(
            {
                "sample_id": row.sample_id,
                "attempt": payload["attempt"],
                "provider": payload["provider"],
                "reasoning_content": payload["provider_raw"]["reasoning_content"],
                "content": payload["provider_raw"]["content"],
            }
        )

    for split in ("train", "validation", "test"):
        is_train = split == "train"
        _write_jsonl(
            stage_root / "prepared" / f"{split}.jsonl",
            prepared_rows if is_train else [],
        )
        _write_jsonl(
            stage_root / "manifests" / f"{split}.jsonl",
            manifest_rows if is_train else [],
        )
        _write_jsonl(
            stage_root / "sft" / f"{split}.jsonl",
            sft_rows if is_train else [],
        )
        _write_jsonl(
            stage_root / "teacher_responses" / f"{split}.jsonl",
            teacher_rows if is_train else [],
        )

    wrapper_path = Path(__file__).resolve()
    target_summary = {
        **summary,
        **evidence_binding,
        "teacher_output_mapping": {
            "reasoning": "json.loads(message.content)['reasoning']",
            "decision": "locally_serialized_canonical_gold",
            "native_reasoning_content": "provenance_only",
        },
        "reasoning_contract": "qualitative_no_numeric_quantities_v2",
        "execution_wrapper": {
            "path": str(wrapper_path),
            "sha256": sha256_file(wrapper_path),
        },
    }
    _write_json(stage_root / "summary.json", target_summary)
    _update_root_summary(output_root, "teacher_targets", target_summary)
    return target_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_teacher_targets_v2(
        output_root=args.output_root.expanduser().resolve(),
        tokenizer_path=args.tokenizer.expanduser().resolve(),
        concurrency=args.concurrency,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROMPT_CONTRACT_FILE", "build_parser", "run_teacher_targets_v2"]
