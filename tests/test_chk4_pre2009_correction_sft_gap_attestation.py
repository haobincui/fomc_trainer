from __future__ import annotations

import json
from pathlib import Path

from jobs.retrain_v2 import attest_chk4_pre2009_correction_sft_gap as gap


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_published_gap_attestation_replays_and_remains_explicitly_post_run():
    result = gap.verify(REPO_ROOT)
    receipt = json.loads(Path(result["path"]).read_text(encoding="utf-8"))

    assert result["sha256"] == (
        "2770fac9ffc58bb1d8a68f018ee3802fd9ad94b9d1a7a51ea45a58f4b2878cbe"
    )
    assert receipt["status"] == "supplemental_not_pre_authorization"
    assert receipt["scope"]["is_not_pre_authorization"] is True
    assert receipt["scope"]["does_not_authorize_rerun"] is True
    assert receipt["scope"]["does_not_authorize_merge_or_grpo"] is True
    assert receipt["scope"][
        "downstream_must_bind_original_and_supplemental_receipts"
    ] is True


def test_gap_attestation_binds_missing_sources_candidates_gpu1_and_abort():
    receipt = gap.build_attestation(REPO_ROOT)

    source_gap = receipt["runtime_source_gap"]
    assert source_gap["source_count"] == 12
    assert source_gap["all_mtime_and_ctime_pre_authorization_and_training"] is True
    assert set(source_gap["sources"]) == set(gap.GAP_RUNTIME_SOURCES)
    assert set(receipt["completed_run"]["candidate_checkpoint_fingerprints"]) == {
        "2",
        "4",
        "6",
    }
    assert receipt["gpu1_evidence"]["physical_device"] == 1
    assert receipt["gpu1_evidence"]["visible_cuda_device_inside_process"] == 0
    assert receipt["aborted_probe_attempt"]["status"] == (
        "aborted_before_generation"
    )
    assert receipt["aborted_probe_attempt"]["completion_rows_landed"] == 0
    assert receipt["aborted_probe_attempt"]["eligible_as_selection_evidence"] is False
