"""Run the 1993--2008 external CHK3 evaluation at N128 x K10.

This is a frozen profile over the durable stochastic generator.  It changes
only the sample/replicate inventory and evaluation identity; WAL, resume,
GPU0 exclusivity, model anchors, generation parameters, and deep validation
remain implemented by ``eval_chk3_stochastic_bootstrap_generation``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from jobs.eval import eval_chk3_stochastic_bootstrap_generation as core
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


ROOT = Path(__file__).resolve().parents[2]
RELEASE_MANIFEST = ROOT / (
    "dataset/processed/retrain_v2/"
    "chk3_minutes_external_holdout_1993_2008_all_regular_v1/release_manifest.json"
)
RELEASE_MANIFEST_SHA256 = (
    "82b045866d9ed6ccbc0d4f00014bffc97694859c2827215309dc8eabe5406937"
)
REPLICATE_SEEDS = (
    20260811,
    21260811,
    22260811,
    23260811,
    24260811,
    25260811,
    26260811,
    27260811,
    28260811,
    29260811,
)


def configure_profile() -> tuple[str, ...]:
    if core._sha256_file(RELEASE_MANIFEST) != RELEASE_MANIFEST_SHA256:
        raise core.StochasticBootstrapGenerationError(
            "external release manifest SHA-256 drift"
        )
    release = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    validate_manifest_integrity(release)
    roster_record = release.get("files", {}).get("official_meeting_roster.jsonl")
    if not isinstance(roster_record, dict):
        raise core.StochasticBootstrapGenerationError("release has no meeting roster")
    roster_path = RELEASE_MANIFEST.parent / str(roster_record.get("path"))
    if core._sha256_file(roster_path) != roster_record.get("sha256"):
        raise core.StochasticBootstrapGenerationError("meeting roster SHA-256 drift")
    meetings = tuple(
        sorted(
            {
                str(json.loads(line)["meeting_start_date"])
                for line in roster_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        )
    )
    if len(meetings) != 128 or meetings[0][:4] != "1993" or meetings[-1][:4] != "2008":
        raise core.StochasticBootstrapGenerationError(
            "external meeting inventory is not the frozen 1993--2008 N128 roster"
        )

    core.EVALUATION_ID = "chk3-external-holdout-1993-2008-n128-k10-v1"
    core.SAMPLE_MANIFEST_SCHEMA_VERSION = (
        "chk3-external-holdout-stochastic-n128-k10-samples-v1"
    )
    core.REPLICATE_SEEDS = REPLICATE_SEEDS
    core.EXPECTED_TEST_ROWS = 128
    core.EXPECTED_MEETING_IDS = meetings
    core.EXPECTED_IDENTITY_ROWS = 0
    core.BOOTSTRAP_SEED = 20260813
    core.MEETING_ID_RE = re.compile(
        r"^chk3-external-((?:19|20)\d{2}-\d{2}-\d{2})-[0-9a-f]+$"
    )
    core.GPU_LOCK_PATH = Path(
        "/tmp/fomc_trainer_chk3_external_holdout_n128_k10_gpu0.lock"
    )
    return meetings


def main() -> int:
    configure_profile()
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
