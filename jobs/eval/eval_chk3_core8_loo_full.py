"""Run the full N128 CHK3 cp318 Core8 leave-one-out evaluation.

This is a versioned full-panel specialization of the already validated N8
infrastructure smoke.  It keeps the same prompt, persistence, GPU0, and
semantic contracts, but includes every regular 1993--2008 meeting in
chronological order: 128 meetings x 17 variants = 2,176 generations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import eval_chk3_core8_loo_smoke as implementation


CONFIG_SCHEMA = "chk3-core8-loo-full-config-v1"
SAMPLE_SCHEMA = "chk3-core8-loo-full-samples-v1"
INPUT_ROW_SCHEMA = "chk3-core8-loo-full-input-row-v1"
RUN_SCHEMA = "chk3-core8-loo-full-generation-run-v1"
STATE_SCHEMA = "chk3-core8-loo-full-generation-state-v1"
SCORE_ROW_SCHEMA = "chk3-core8-loo-full-score-row-v1"
SCORE_SCHEMA = "chk3-core8-loo-full-score-manifest-v1"
EVALUATION_ID = "chk3-cp318-core8-loo-full-n128-1993-2008-v1"
EXPECTED_MEETINGS = 128
EXPECTED_ROWS = 2176
MODEL_LABEL = "chk3-cp318-exact-merged-core8-loo-full-n128"
DEFAULT_CONFIG = (
    implementation.ROOT
    / "configs/main/chk3_cp318_core8_loo_full_n128_1993_2008_v1.json"
)
GPU_LOCK = Path("/tmp/fomc_trainer_chk3_core8_loo_full_gpu0.lock")
EVALUATION_SCOPE = "full_n128_raw_loo_descriptive"
PREPARE_LIMITATIONS = (
    "This full panel covers all 128 sealed regular 1993--2008 meetings.",
    "The primary reference is a deterministic concatenation of eight source-grounded Minutes-style references, not an official Minutes excerpt.",
    "The result measures CHK3 stage-local indicator sensitivity, not causal contribution.",
)
RUN_LIMITATIONS = (
    "The full N128 run is descriptive; inferential uncertainty requires a separate meeting-clustered analysis.",
    "The generation path uses 4-bit NF4 inference over the exact-merged cp318 artifact.",
)
SCORE_LIMITATIONS = (
    "Raw paired MPNet and BERTScore deltas are primary; generation gates are diagnostics only.",
    "The source-grounded concatenated reference is not an official Minutes excerpt.",
    "Positive target-relative delta does not identify a causal indicator contribution.",
)
FAILURE_ROW_SCHEMA = "chk3-core8-loo-full-failure-v1"


def select_all_meetings(
    rows: Sequence[Mapping[str, Any]], bins: Sequence[Sequence[int]]
) -> list[str]:
    if [list(values) for values in bins] != [[1993, 2008]]:
        raise implementation.Core8LooSmokeError(
            "full Core8 inventory requires the sealed 1993--2008 range"
        )
    meetings: dict[str, str] = {}
    for row in rows:
        meeting_id = row.get("meeting_id")
        date = row.get("meeting_start_date")
        if not isinstance(meeting_id, str) or not isinstance(date, str):
            raise implementation.Core8LooSmokeError(
                "Core8 row has no meeting identity"
            )
        if not ("1993-01-01" <= date <= "2008-12-31"):
            raise implementation.Core8LooSmokeError(
                "meeting lies outside the sealed 1993--2008 range"
            )
        if meeting_id in meetings and meetings[meeting_id] != date:
            raise implementation.Core8LooSmokeError(
                "meeting date drift in Core8 inventory"
            )
        meetings[meeting_id] = date
    selected = sorted(meetings, key=lambda meeting_id: (meetings[meeting_id], meeting_id))
    if len(selected) != EXPECTED_MEETINGS:
        raise implementation.Core8LooSmokeError(
            f"full Core8 inventory is not N{EXPECTED_MEETINGS}"
        )
    return selected


def configure_implementation() -> None:
    implementation.CONFIG_SCHEMA = CONFIG_SCHEMA
    implementation.SAMPLE_SCHEMA = SAMPLE_SCHEMA
    implementation.INPUT_ROW_SCHEMA = INPUT_ROW_SCHEMA
    implementation.RUN_SCHEMA = RUN_SCHEMA
    implementation.STATE_SCHEMA = STATE_SCHEMA
    implementation.SCORE_ROW_SCHEMA = SCORE_ROW_SCHEMA
    implementation.SCORE_SCHEMA = SCORE_SCHEMA
    implementation.EVALUATION_ID = EVALUATION_ID
    implementation.EXPECTED_MEETINGS = EXPECTED_MEETINGS
    implementation.EXPECTED_ROWS = EXPECTED_ROWS
    implementation.MODEL_LABEL = MODEL_LABEL
    implementation.DEFAULT_CONFIG = DEFAULT_CONFIG
    implementation.GPU_LOCK = GPU_LOCK
    implementation.EVALUATION_SCOPE = EVALUATION_SCOPE
    implementation.PREPARE_LIMITATIONS = PREPARE_LIMITATIONS
    implementation.RUN_LIMITATIONS = RUN_LIMITATIONS
    implementation.SCORE_LIMITATIONS = SCORE_LIMITATIONS
    implementation.FAILURE_ROW_SCHEMA = FAILURE_ROW_SCHEMA
    implementation._select_meetings = select_all_meetings


def main() -> int:
    configure_implementation()
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
