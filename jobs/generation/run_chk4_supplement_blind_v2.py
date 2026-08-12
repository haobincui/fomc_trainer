"""Run the versioned qualitative blind-decision audit for the chk4 supplement."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from jobs.generation.generate_chk4_supplement import (
    BLIND_SCHEMA,
    DECISION_MAX_TOKENS,
    DEFAULT_CONCURRENCY,
    DEFAULT_OUTPUT_ROOT,
    StageOutputError,
    StageRow,
    _decision_rows,
    _evidence_profile_binding,
    _load_tokenizer,
    _run_stage,
    _update_root_summary,
    _validate_blind,
    _write_json,
    _write_jsonl,
    qualitative_reasoning_has_number_or_date,
)
from jobs.generation.generate_chk4_sft_targets import (
    DEFAULT_TOKENIZER,
    TeacherResponse,
    sha256_file,
)


BLIND_V2_SYSTEM_PROMPT = """\
You are auditing whether a target-neutral pre-meeting analysis is sufficient
for one FOMC policy decision. Use only the supplied analysis and weigh the
evidence under the maximum-employment and price-stability objectives.

Do not use outside or remembered historical information, infer the meeting
identity, or claim that the Committee actually took an action. Return exactly
one JSON object with keys reasoning, direction, and magnitude_bp. Reasoning
must be one grounded qualitative paragraph under 180 words with no digits,
dates, percentages, basis-point amounts, or spelled-out numeric quantities.
Put the numeric decision only in magnitude_bp. Direction must be cut, hold, or
hike. Hold requires magnitude 0; cut and hike require 25, 50, 75, or 100.
Output no other fields or content.
"""

BLIND_V2_REPAIR_SYSTEM_PROMPT = """\
Regenerate the decision JSON using only the supplied pre-meeting analysis and
silently correct the format or grounding failure. Return exactly the keys
reasoning, direction, and magnitude_bp. Reasoning must be one grounded
qualitative paragraph under 180 words with no digits, dates, percentages,
basis-point amounts, or spelled-out numeric quantities. Put the numeric
decision only in magnitude_bp. Direction must be cut, hold, or hike; hold
requires zero and cut or hike requires 25, 50, 75, or 100. This is a bounded
format repair attempt.
"""

PROMPT_CONTRACT_FILE = "prompt_contract.qualitative-v2.json"


def _validate_blind_v2(
    row: StageRow, response: TeacherResponse, tokenizer: Any
) -> dict[str, Any]:
    """Keep the original blind contract and additionally forbid quantities."""

    target = _validate_blind(row, response, tokenizer)
    if qualitative_reasoning_has_number_or_date(str(target["reasoning"])):
        raise StageOutputError(("blind_reasoning_contains_number_or_date",))
    return target


def run_blind_predictions_v2(
    *,
    output_root: Path,
    tokenizer_path: Path,
    concurrency: int,
    resume: bool,
    backend: Any | None = None,
    environment: Mapping[str, str] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    rows = _decision_rows(output_root, include_gold=False)
    tokenizer = tokenizer or _load_tokenizer(tokenizer_path)
    stage_root = output_root / "blind_predictions"
    accepted, summary = _run_stage(
        stage="blind_predictions",
        rows=rows,
        output_root=output_root,
        stage_root=stage_root,
        system_prompt=BLIND_V2_SYSTEM_PROMPT,
        repair_prompt=BLIND_V2_REPAIR_SYSTEM_PROMPT,
        max_tokens=DECISION_MAX_TOKENS,
        tokenizer=tokenizer,
        validator=_validate_blind_v2,
        concurrency=concurrency,
        resume=resume,
        backend=backend,
        environment=environment,
        prompt_contract_file=PROMPT_CONTRACT_FILE,
        repair_attempts=2,
    )
    wrapper_path = Path(__file__).resolve()
    summary = {
        **summary,
        **_evidence_profile_binding(output_root),
        "execution_wrapper": {
            "path": str(wrapper_path),
            "sha256": sha256_file(wrapper_path),
        },
        "reasoning_contract": "qualitative_no_numeric_quantities_v2",
    }
    _write_json(stage_root / "summary.json", summary)
    _update_root_summary(output_root, "blind_predictions", summary)

    predictions = []
    teacher_rows = []
    for row in rows:
        payload = accepted[row.sample_id]
        predictions.append(
            {
                "schema_version": BLIND_SCHEMA,
                "sample_id": row.sample_id,
                **payload["target"],
                "analysis_sha256": row.input_sha256,
                "contract_sha256": summary["contract_sha256"],
            }
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
    _write_jsonl(stage_root / "predictions.jsonl", predictions)
    _write_jsonl(stage_root / "teacher_responses.jsonl", teacher_rows)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_blind_predictions_v2(
        output_root=args.output_root.resolve(),
        tokenizer_path=args.tokenizer.resolve(),
        concurrency=args.concurrency,
        resume=args.resume,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
