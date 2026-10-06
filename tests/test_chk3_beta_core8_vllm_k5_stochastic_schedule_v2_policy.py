from __future__ import annotations

from pathlib import Path

import pytest

from jobs.eval import (
    remediate_chk3_beta_core8_vllm_k5_dual_dp1_stochastic_schedule_v2 as amendment,
)
from jobs.eval import (
    seal_chk3_beta_core8_vllm_k5_dual_dp1_stochastic_schedule_v2_suite as suite,
)
from jobs.eval import (
    select_chk3_beta_core8_vllm_k5_dual_dp1_stochastic_schedule_v2 as selector,
)


def _row(index: int, *, output: str, shard_id: int | None = None) -> dict:
    shard = index % 2 if shard_id is None else shard_id
    return {
        "model_id": "chk1",
        "sample_id": f"sample-{index}",
        "replicate_id": 0,
        "replicate_seed": 20260811,
        "row_seed": 1000 + index,
        "seed": 1000 + index,
        "absolute_case_index": index,
        "absolute_chunk_id": 0,
        "shard_id": shard,
        "prompt_sha256": f"prompt-{index}",
        "prompt_token_ids_sha256": f"tokens-{index}",
        "prompt_token_count": 10,
        "input_token_count": 10,
        "source_prompt_sha256": f"source-prompt-{index}",
        "source_analysis_sha256": f"analysis-{index}",
        "reference_minutes_sha256": f"minutes-{index}",
        "generated_token_ids": [index],
        "generated_text": output,
        "answer": output,
        "finish_reason": "eos",
        "completion_sha256": output,
        "answer_sha256": output,
        "generated_token_ids_sha256": output,
        "status": "ok",
        "vllm_raw_finish_reason": "stop",
        "hit_eos": True,
        "cap_reached": False,
        "input_truncated": False,
    }


def test_speed_argmax_is_pure_speed_and_fixed16() -> None:
    rates = {8: 496.8615176827086, 12: 628.02361954562, 16: 704.6470564170767}
    assert selector._speed_argmax(rates) == 16
    assert selector._speed_argmax({8: 1.0, 12: 3.0, 16: 2.0}) == 12
    with pytest.raises(selector.StochasticScheduleSelectionError):
        selector._speed_argmax({8: 1.0, 12: 2.0})


def test_token_volume_gate_is_five_percent() -> None:
    assert selector._token_volume_ratio([21630, 21861]) == pytest.approx(21861 / 21630)
    assert selector._token_volume_ratio([21630, 21861]) <= 1.05
    assert selector._token_volume_ratio([100, 106]) > 1.05
    with pytest.raises(selector.StochasticScheduleSelectionError):
        selector._token_volume_ratio([0, 1])


def test_same_config_token_differences_are_diagnostic_not_gate() -> None:
    original = [_row(index, output=f"a-{index}") for index in range(40)]
    exact_indexes = {index for index in range(40) if index % 2 == 1} | {0, 2}
    replay = [
        _row(index, output=f"a-{index}" if index in exact_indexes else f"b-{index}")
        for index in range(40)
    ]
    diagnostic = selector._exact_output_diagnostic(original, replay)
    assert diagnostic["exact_output_rows"] == 22
    assert diagnostic["different_output_rows"] == 18
    assert diagnostic["exact_output_identity_required"] is False
    assert diagnostic["by_shard"]["shard0"]["exact_output_rows"] == 2
    assert diagnostic["by_shard"]["shard1"]["exact_output_rows"] == 20


def test_amendment_diagnostic_requires_exact_input_seed_tuple() -> None:
    left = [_row(index, output=f"a-{index}") for index in range(4)]
    right = [_row(index, output=f"b-{index}") for index in range(4)]
    diagnostic = amendment._diagnostic(left, right)
    assert diagnostic["exact_output_rows"] == 0
    assert diagnostic["blocking"] is False
    right[0]["row_seed"] += 1
    with pytest.raises(amendment.StochasticScheduleAmendmentError):
        amendment._diagnostic(left, right)


def test_normal_finish_gate_rejects_cap_or_error() -> None:
    rows = [_row(index, output="x") for index in range(40)]
    assert selector._normal_finish_count(rows) == 40
    rows[0]["cap_reached"] = True
    assert selector._normal_finish_count(rows) == 39
    rows[1]["status"] = "error"
    assert selector._normal_finish_count(rows) == 38


def test_formal_authorization_has_exact_stable_shape() -> None:
    manifest_binding = {"path": "/sealed/smoke.json", "sha256": "a" * 64}
    speed = {"path": "/sealed/selection.json", "sha256": "b" * 64}
    diagnostic = {
        "classification": "schedule_sensitive_seeded_sampling",
        "exact_token_identity_is_acceptance_gate": False,
    }
    loaded = {
        "manifest_binding": manifest_binding,
        "manifest": {
            "speed_selection": speed,
            "schedule_sensitivity_diagnostic": diagnostic,
            "generation_gates": dict(suite.GENERATION_GATES),
        },
    }
    authorization = suite._formal_authorization_from_smoke_loaded(loaded)
    assert authorization == {
        "official_smoke_suite": manifest_binding,
        "speed_selection": speed,
        "schedule_sensitivity_diagnostic": diagnostic,
        "generation_gates": suite.GENERATION_GATES,
    }


def test_v4_paths_do_not_overlap_historical_v3_outputs() -> None:
    assert amendment.V3_BENCHMARK_ROOT not in amendment.FRESH_V4_ROOTS
    assert all("v3_dual_dp1" not in path.name for path in amendment.FRESH_V4_ROOTS)
    assert amendment.DEFAULT_RECEIPT != amendment.v3_remediation.DEFAULT_RECEIPT
    assert not Path(selector.__file__).samefile(Path(selector.v3_runner.__file__))
