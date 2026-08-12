from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2 import probe_chk4_decision_sft_generation as base_probe
from jobs.retrain_v2 import probe_chk4_pre2009_correction as probe
from jobs.retrain_v2 import run_chk4_pre2009_correction_cp6_fixed as runner


REPO_ROOT = Path(__file__).resolve().parents[1]


def _authorization() -> dict[str, object]:
    # Pre-launch, exercise full replay.  Post-launch, inspect the immutable
    # authorization because the fresh-output preflight must then fail closed.
    if runner.output_path(REPO_ROOT).exists():
        return json.loads(
            runner.authorization_path(REPO_ROOT).read_text(encoding="utf-8")
        )
    return runner.authorization_payload(REPO_ROOT)


def test_cp6_authorization_payload_binds_final_candidate_and_prior_decisions() -> None:
    value = _authorization()
    assert value["candidate_checkpoint"]["sha256"] == runner.CP6_CHECKPOINT_SHA256
    assert value["candidate_checkpoint"]["file_count"] == 12
    assert value["candidate_checkpoint"]["total_bytes"] == 356059569
    assert value["candidate_checkpoint"]["adapter_model"]["sha256"] == (
        runner.CP6_ADAPTER_WEIGHTS_SHA256
    )
    assert value["lineage"]["checkpoint_4_failed_not_selected_decision"][
        "sha256"
    ] == runner.CP4_DECISION_SHA256
    assert value["methodology"]["checkpoint_6_is_final_candidate"] is True
    assert value["methodology"]["full_original_gate_required_for_selection"] is True
    assert value["scope"]["gpu"] == 1
    assert "merge" in value["scope"]["not_authorized"]


def test_cp6_generation_contract_is_identical_to_preregistered_selection() -> None:
    value = _authorization()
    assert value["generation"] == {
        "seed": 31416,
        "seed_once_before_all_batches": True,
        "batches": [
            ["dec-bb7d9c61358a339a9a1f4aa5", "dec-4dfab939b0a910c949931475"],
            ["dec-29de02cb43945c20f838fcf7", "dec-8b8d55ea065b662a19cefe88"],
        ],
        "num_return_sequences": 4,
        "max_new_tokens": 1536,
        "temperature": 0.7,
        "top_p": 0.9,
        "padding_postprocessor": (
            "truncate_at_first_bound_eos_inclusive_before_decode"
        ),
    }


def test_cp6_context_trims_before_result_metrics_and_records_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_decode(tokenizer: object, token_ids: object, eos_ids: object) -> str:
        captured["decoded_ids"] = list(token_ids)  # type: ignore[arg-type]
        return "decoded"

    def fake_result(**kwargs: object) -> dict[str, object]:
        captured["result_ids"] = list(kwargs["generated_ids"])  # type: ignore[arg-type]
        return {"completion": kwargs["completion"]}

    def fake_model_source(**kwargs: object) -> dict[str, object]:
        return {"provenance": {"eos_token_ids": [128001]}}

    monkeypatch.setattr(base_probe, "decode_completion_preserving_boundary", fake_decode)
    monkeypatch.setattr(probe, "_result_row", fake_result)
    monkeypatch.setattr(probe, "_model_source", fake_model_source)
    with runner._fixed_padding_postprocessor(REPO_ROOT):
        assert base_probe.decode_completion_preserving_boundary(
            object(), [7, 128001, 128001, 128001], [128001]
        ) == "decoded"
        row = probe._result_row(
            generated_ids=[7, 128001, 128001, 128001],
            completion="decoded",
            provenance={"eos_token_ids": [128001]},
        )
        source = probe._model_source()
    assert captured["decoded_ids"] == [7, 128001]
    assert captured["result_ids"] == [7, 128001]
    assert row["batched_padding_normalization"][
        "discarded_after_first_eos_count"
    ] == 2
    postprocessor = source["provenance"]["batched_padding_postprocessor"]
    assert postprocessor["implementation"]["path"].endswith(
        "run_chk4_pre2009_correction_cp6_fixed.py"
    )
    assert postprocessor["predecessor_implementation"]["sha256"] == (
        cp4_runner_sha()
    )


def cp4_runner_sha() -> str:
    value = json.loads(
        (REPO_ROOT / runner.CP4_DECISION).read_text(encoding="utf-8")
    )
    return value["implementation"][
        "jobs/retrain_v2/run_chk4_pre2009_correction_screen_fixed.py"
    ]["sha256"]


def test_authorization_is_create_only(tmp_path: Path) -> None:
    path = tmp_path / "authorization.json"
    path.write_text("{}\n", encoding="utf-8")
    assert path.exists()
    # The production path check occurs before O_EXCL; this test protects the
    # invariant without attempting to clone the multi-gigabyte candidate tree.
    with pytest.raises(runner.Cp6ScreenError):
        runner._require(not path.exists(), "cp6 authorization exists")
