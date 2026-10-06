from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jobs.eval import score_chk3_core8_loo_vllm_k5 as subject
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


def _matrices(value: float = 0.02) -> dict[tuple[str, str, str], np.ndarray]:
    return {
        (topic, arm, metric): np.full(
            (subject.EXPECTED_MEETINGS, subject.EXPECTED_REPLICATES),
            value,
            dtype=np.float64,
        )
        for topic in subject.TOPICS
        for arm in subject.ARMS
        for metric in subject.METRICS
    }


def _meeting_ids() -> list[str]:
    return [f"meeting-{index:03d}" for index in range(subject.EXPECTED_MEETINGS)]


def _write_sealed(path: Path, payload: dict) -> tuple[dict, str]:
    sealed = seal_manifest(payload)
    path.write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sealed, subject._sha256_file(path)


def _minimal_plan() -> dict:
    return {
        "schema_version": subject.PLAN_SCHEMA,
        "status": "complete",
        "evaluation_id": subject.SCORE_EVALUATION_ID,
        "coverage": {
            "rows": subject.EXPECTED_ROWS,
            "empty_answer_rows": subject.EXPECTED_EMPTY_ANSWER_ROWS,
            "tuple_key_sha256": "a" * 64,
        },
        "inputs": {
            "generation_manifest": {"path": "/generation"},
            "cohort": {"path": "/cohort"},
            "semantic_manifest": {"path": "/semantic"},
            "canonical_generations": {"path": "/rows"},
            "historical_k1_anchor": {
                "score_manifest": {"path": "/k1-score"},
                "meeting_bootstrap_manifest": {"path": "/k1-bootstrap"},
            },
        },
        "bootstrap_contract": {
            "draws": subject.BOOTSTRAP_DRAWS,
            "seed": subject.BOOTSTRAP_SEED,
            "confidence": subject.CONFIDENCE,
            "meeting_resampling": "128_with_replacement",
            "replicate_resampling": "5_with_replacement_per_sampled_meeting_occurrence",
            "paired_within_replicate": True,
            "shared_indices_across_all_32_cells": True,
            "primary_interval": "hierarchical_percentile",
            "meeting_only_interval": "fixed_K5_meeting_resampling_sensitivity",
        },
        "k_adequacy_rule": {
            "status": subject.K_ADEQUACY_FREEZE_STATUS,
            "stochastic_prefixes": subject.K_PREFIX_POLICY,
            "k5_minus_k4_point_drift_threshold": (
                subject.K5_K4_POINT_DRIFT_THRESHOLD
            ),
            "k5_minus_k4_ci_endpoint_drift_threshold": (
                subject.K5_K4_CI_ENDPOINT_DRIFT_THRESHOLD
            ),
            "leave_one_replicate_max_drift_threshold": (
                subject.LORO_MAX_DRIFT_THRESHOLD
            ),
            "decoding_variance_share_threshold": (
                subject.DECODING_VARIANCE_SHARE_THRESHOLD
            ),
            "increase_k_if": list(subject.K_INCREASE_RULES),
        },
        "sources": {role: {"path": f"/{role}"} for role in subject.SOURCE_ROLES},
    }


def test_frozen_raw_k5_and_adequacy_contract() -> None:
    assert subject.EXPECTED_ROWS == 10_880
    assert subject.EXPECTED_MEETINGS == 128
    assert subject.EXPECTED_REPLICATES == 5
    assert subject.BOOTSTRAP_DRAWS == 10_000
    assert subject.BOOTSTRAP_SEED == 20_260_817
    assert subject.K5_K4_POINT_DRIFT_THRESHOLD == 0.005
    assert subject.K5_K4_CI_ENDPOINT_DRIFT_THRESHOLD == 0.010
    assert subject.LORO_MAX_DRIFT_THRESHOLD == 0.010
    assert subject.DECODING_VARIANCE_SHARE_THRESHOLD == 0.20
    assert len(subject.TOPICS) * len(subject.ARMS) * len(subject.METRICS) == 32
    assert subject.SCORE_EVALUATION_ID.endswith("raw-semantic-b10000-v2")
    assert subject.DEFAULT_OUTPUT_DIR.name == "score_raw_semantic_dual_gpu_b10000_v2"
    launcher = (
        subject.ROOT / "run/score_chk3_core8_loo_vllm_k5_dual_gpu.sh"
    ).read_text(encoding="utf-8")
    assert "score_raw_semantic_dual_gpu_b10000_v2" in launcher
    assert "score_raw_semantic_dual_gpu_b10000_v1" not in launcher


def test_shared_hierarchical_plan_is_reproducible_and_prefix_coupled() -> None:
    first = subject.make_bootstrap_plan(
        meeting_ids=_meeting_ids(), draws=40, seed=19
    )
    same = subject.make_bootstrap_plan(
        meeting_ids=_meeting_ids(), draws=40, seed=19
    )
    changed = subject.make_bootstrap_plan(
        meeting_ids=_meeting_ids(), draws=40, seed=20
    )
    assert first.sha256 == same.sha256
    assert first.sha256 != changed.sha256
    np.testing.assert_array_equal(first.meeting_indices, same.meeting_indices)
    assert first.replicate_indices.shape == (40, 128, 5)
    for prefix in range(1, 6):
        values = first.prefix_replicate_indices[prefix]
        assert values.shape == (40, 128, prefix)
        assert int(values.min()) >= 0
        assert int(values.max()) < prefix
    assert np.all(first.prefix_replicate_indices[1] == 0)
    assert first.metadata["paired_indices"] is True


def test_constant_panel_has_exact_point_and_degenerate_intervals() -> None:
    computation = subject.compute_bootstrap(
        matrices=_matrices(0.02),
        meeting_ids=_meeting_ids(),
        draws=100,
        seed=29,
    )
    validate_manifest_integrity(computation.results)
    cell = computation.results["cells"][subject.TOPICS[0]][subject.ARMS[0]][
        subject.METRICS[0]
    ]
    assert math.isclose(cell["point_estimate_mean_delta"], 0.02)
    assert math.isclose(cell["hierarchical_bootstrap"]["ci_lower"], 0.02)
    assert math.isclose(cell["hierarchical_bootstrap"]["ci_upper"], 0.02)
    assert cell["positive_meetings"] == 128
    adequacy = computation.results["k_adequacy"]
    assert adequacy["increase_k_recommended"] is False
    assert adequacy["aggregate_status"] == "k5_adequate_for_aggregate_estimand"
    prefix = cell["k_adequacy"]["cumulative_fresh_stochastic_prefixes"]
    assert list(prefix) == ["k1", "k2", "k3", "k4", "k5"]
    assert all(
        record["historical_greedy_k1_included"] is False
        for record in prefix.values()
    )


def test_prefix_point_drift_triggers_frozen_k_increase_rule() -> None:
    matrices = _matrices(0.0)
    target = (subject.TOPICS[0], subject.ARMS[0], subject.METRICS[0])
    matrices[target][:, 4] = 0.10
    result = subject.compute_bootstrap(
        matrices=matrices,
        meeting_ids=_meeting_ids(),
        draws=200,
        seed=31,
    ).results
    cell = result["cells"][target[0]][target[1]][target[2]]["k_adequacy"]
    assert math.isclose(cell["k5_minus_k4_point_drift"], 0.02)
    assert "k5_minus_k4_point_drift_above_threshold" in cell[
        "increase_k_reasons"
    ]
    assert result["k_adequacy"]["increase_k_recommended"] is True


def test_every_cell_and_prefix_uses_the_same_nested_indices() -> None:
    matrices = _matrices(0.0)
    first = (subject.TOPICS[0], subject.ARMS[0], subject.METRICS[0])
    second = (subject.TOPICS[1], subject.ARMS[1], subject.METRICS[1])
    values = np.arange(128 * 5, dtype=np.float64).reshape(128, 5) / 1000.0
    matrices[first] = values
    matrices[second] = 2.0 * values
    computation = subject.compute_bootstrap(
        matrices=matrices,
        meeting_ids=_meeting_ids(),
        draws=50,
        seed=37,
    )
    np.testing.assert_allclose(
        computation.cell_draws[second], 2.0 * computation.cell_draws[first]
    )
    for prefix in range(1, 6):
        np.testing.assert_allclose(
            computation.prefix_cell_draws[(*second, prefix)],
            2.0 * computation.prefix_cell_draws[(*first, prefix)],
        )


def test_bootstrap_draw_ledger_persists_every_prefix_index_and_value(
    tmp_path: Path,
) -> None:
    computation = subject.compute_bootstrap(
        matrices=_matrices(0.02),
        meeting_ids=_meeting_ids(),
        draws=2,
        seed=43,
    )
    path = tmp_path / "draws.jsonl"
    subject._write_bootstrap_draws(path, computation)
    rows = list(subject._iter_jsonl(path))
    assert len(rows) == 2
    first = rows[0]
    assert first["index_plan_sha256"] == computation.plan.sha256
    assert list(first["replicate_indices_by_stochastic_prefix"]) == [
        "k1",
        "k2",
        "k3",
        "k4",
        "k5",
    ]
    assert first["replicate_indices"] == first[
        "replicate_indices_by_stochastic_prefix"
    ]["k5"]
    cell = first["cell_mean_deltas"][subject.TOPICS[0]][subject.ARMS[0]][
        subject.METRICS[0]
    ]
    assert list(cell["stochastic_prefix_hierarchical_mean_delta"]) == [
        "k1",
        "k2",
        "k3",
        "k4",
        "k5",
    ]
    for prefix in range(1, 6):
        assert math.isclose(
            cell["stochastic_prefix_hierarchical_mean_delta"][f"k{prefix}"],
            computation.prefix_cell_draws[
                (
                    subject.TOPICS[0],
                    subject.ARMS[0],
                    subject.METRICS[0],
                    prefix,
                )
            ][0],
        )


@pytest.mark.parametrize(
    "thresholds,expected_reason",
    [
        (
            {
                "k5_minus_k4_point_drift_threshold": 1.0,
                "k5_minus_k4_ci_endpoint_drift_threshold": 0.000001,
                "leave_one_replicate_max_drift_threshold": 1.0,
                "decoding_variance_share_threshold": 10.0,
            },
            "k5_minus_k4_ci_endpoint_drift_above_threshold",
        ),
        (
            {
                "k5_minus_k4_point_drift_threshold": 1.0,
                "k5_minus_k4_ci_endpoint_drift_threshold": 1.0,
                "leave_one_replicate_max_drift_threshold": 0.000001,
                "decoding_variance_share_threshold": 10.0,
            },
            "leave_one_replicate_max_drift_above_threshold",
        ),
        (
            {
                "k5_minus_k4_point_drift_threshold": 1.0,
                "k5_minus_k4_ci_endpoint_drift_threshold": 1.0,
                "leave_one_replicate_max_drift_threshold": 1.0,
                "decoding_variance_share_threshold": 0.000001,
            },
            "decoding_variance_share_above_threshold",
        ),
    ],
)
def test_each_adequacy_diagnostic_can_trigger_independently(
    thresholds: dict[str, float], expected_reason: str
) -> None:
    matrices = _matrices(0.0)
    target = (subject.TOPICS[0], subject.ARMS[0], subject.METRICS[0])
    matrices[target][:, 4] = 0.06
    result = subject.compute_bootstrap(
        matrices=matrices,
        meeting_ids=_meeting_ids(),
        draws=100,
        seed=41,
        adequacy_thresholds=thresholds,
    ).results
    reasons = result["cells"][target[0]][target[1]][target[2]]["k_adequacy"][
        "increase_k_reasons"
    ]
    assert expected_reason in reasons


def _generation_projection(*, core_valid: bool) -> dict[str, object]:
    failures = [] if core_valid else ["numeric_multiset_not_preserved"]
    return {
        "absolute_case_index": 0,
        "meeting_id": "meeting-0",
        "meeting_rank": 0,
        "variant_rank": 0,
        "arm": "full",
        "intervention_topic": None,
        "replicate_id": 0,
        "replicate_seed": subject.REPLICATE_SEEDS[0],
        "row_seed": 17,
        "sample_id": "sample-0",
        "answer": "nonempty answer",
        "completion_sha256": "a" * 64,
        "answer_sha256": "b" * 64,
        "generated_token_ids_sha256": "c" * 64,
        "reference_minutes_sha256": "d" * 64,
        "finish_reason": "eos",
        "input_truncated": False,
        "content_tokens": 12,
        "full_token_4gram_repetition": 0.1,
        "tail_token_4gram_repetition": 0.1,
        "gate": {
            "delivery_valid": True,
            "native_structure_valid": True,
            "numeric_multiset_preserved": core_valid,
            "date_set_preserved": True,
            "degeneration_free": True,
            "preregistered_core_valid": core_valid,
            "preregistered_core_failures": failures,
        },
    }


def _metric_row(worker: str, source: dict[str, object]) -> dict[str, object]:
    scores = (
        {
            "bertscore_precision": 0.41,
            "bertscore_recall": 0.42,
            "bertscore_f1": 0.43,
        }
        if worker == "bertscore"
        else {"mpnet_cosine": 0.37}
    )
    return {
        "schema_version": subject.METRIC_ROW_SCHEMA,
        "metric_worker": worker,
        "source_generation_row": 1,
        "tuple_key": subject._tuple_key(source),
        "sample_id": source["sample_id"],
        "answer_sha256": source["answer_sha256"],
        "reference_minutes_sha256": source["reference_minutes_sha256"],
        "scores": scores,
    }


def test_generation_gate_failure_is_diagnostic_and_never_zeroes_raw_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "EXPECTED_ROWS", 1)
    monkeypatch.setattr(subject, "EXPECTED_EMPTY_ANSWER_ROWS", 0)
    source = _generation_projection(core_valid=False)
    rows, failures = subject.build_scored_rows(
        generation_rows=[source],
        bert_rows=[_metric_row("bertscore", source)],
        mpnet_rows=[_metric_row("mpnet", source)],
    )
    assert rows[0]["raw_semantic_scores"]["mpnet_cosine"] == 0.37
    assert rows[0]["raw_semantic_scores"]["bertscore_f1"] == 0.43
    assert rows[0]["primary_loo_policy"] == {
        "included": True,
        "gate_filter_applied": False,
        "gate_zero_penalty_applied": False,
    }
    assert failures[0]["included_in_primary_loo"] is True
    assert failures[0]["effect_on_primary_score"] == "none_diagnostic_only"


def test_metric_alignment_rejects_a_different_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "EXPECTED_ROWS", 1)
    monkeypatch.setattr(subject, "EXPECTED_EMPTY_ANSWER_ROWS", 0)
    source = _generation_projection(core_valid=True)
    bert = _metric_row("bertscore", source)
    bert["tuple_key"] = {**bert["tuple_key"], "row_seed": 99}
    with pytest.raises(subject.LooK5ScoreError, match="alignment drift"):
        subject.build_scored_rows(
            generation_rows=[source],
            bert_rows=[bert],
            mpnet_rows=[_metric_row("mpnet", source)],
        )


@pytest.mark.parametrize(
    "metric,vectors,changed",
    [
        (
            "bertscore",
            {
                "bertscore_precision": [0.0, 0.2],
                "bertscore_recall": [0.0, 0.3],
                "bertscore_f1": [0.0, 0.4],
            },
            "bertscore_f1",
        ),
        ("mpnet", {"mpnet_cosine": [0.0, 0.5]}, "mpnet_cosine"),
    ],
)
def test_empty_candidate_all_raw_metric_invariant(
    metric: str,
    vectors: dict[str, list[float]],
    changed: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "EXPECTED_EMPTY_ANSWER_ROWS", 1)
    rows = [{"answer": ""}, {"answer": "nonempty"}]
    assert (
        subject._assert_empty_candidate_scores(
            metric=metric, rows=rows, vectors=vectors
        )
        == 1
    )
    invalid = {name: list(values) for name, values in vectors.items()}
    invalid[changed][0] = 0.01
    with pytest.raises(subject.LooK5ScoreError, match="empty candidate"):
        subject._assert_empty_candidate_scores(
            metric=metric, rows=rows, vectors=invalid
        )


def test_finalize_row_invariant_rejects_nonzero_empty_bert_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "EXPECTED_ROWS", 1)
    monkeypatch.setattr(subject, "EXPECTED_EMPTY_ANSWER_ROWS", 1)
    source = _generation_projection(core_valid=False)
    source["answer"] = ""
    bert = _metric_row("bertscore", source)
    bert["scores"]["bertscore_precision"] = 0.1
    mpnet = _metric_row("mpnet", source)
    mpnet["scores"]["mpnet_cosine"] = 0.0
    with pytest.raises(subject.LooK5ScoreError, match="empty candidate"):
        subject.build_scored_rows(
            generation_rows=[source], bert_rows=[bert], mpnet_rows=[mpnet]
        )


@pytest.mark.parametrize(
    ("physical_gpu_index", "uuid", "pci_bus_id", "torch_pci_bus"),
    [
        (
            0,
            "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
            "00000000:21:00.0",
            33,
        ),
        (
            1,
            "GPU-0d2b981c-7a91-ad4c-7015-89445a0283a1",
            "00000000:e1:00.0",
            225,
        ),
    ],
)
def test_gpu_validation_accepts_torch26_integer_bus_for_both_real_gpus(
    monkeypatch: pytest.MonkeyPatch,
    physical_gpu_index: int,
    uuid: str,
    pci_bus_id: str,
    torch_pci_bus: int,
) -> None:
    properties = SimpleNamespace(
        uuid=uuid.lower(),
        pci_domain_id=0,
        pci_bus_id=torch_pci_bus,
        pci_device_id=0,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: properties,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(physical_gpu_index))
    monkeypatch.setattr(
        subject.gpu_runtime,
        "_physical_gpu_identity",
        lambda index: {
            "physical_gpu_index": index,
            "uuid": uuid,
            "pci_bus_id": pci_bus_id,
        },
    )
    observed_torch, observed = subject._validate_metric_environment(
        physical_gpu_index
    )
    assert observed_torch is fake_torch
    assert observed == {
        "logical_cuda_index": 0,
        "physical_gpu_index": physical_gpu_index,
        "uuid": uuid,
        "pci_bus_id": pci_bus_id,
    }


def test_gpu_validation_normalises_full_string_pci_address() -> None:
    identity = {
        "physical_gpu_index": 0,
        "uuid": "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
        "pci_bus_id": "00000000:21:00.0",
    }
    properties = SimpleNamespace(
        uuid="ba3a7f4c74c84922364d031c5187624c",
        pci_bus_id="0000:21:00.0",
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: properties,
        )
    )
    assert subject._verify_visible_cuda_device(fake_torch, identity) == {
        "logical_cuda_index": 0,
        **identity,
    }


@pytest.mark.parametrize(
    ("observed_uuid", "observed_domain", "observed_pci", "observed_device", "error"),
    [
        (
            "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff",
            0,
            33,
            0,
            "UUID does not match",
        ),
        (
            "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
            0,
            34,
            0,
            "PCI domain/bus/device does not match",
        ),
        (
            "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
            1,
            33,
            0,
            "PCI domain/bus/device does not match",
        ),
        (
            "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
            0,
            33,
            1,
            "PCI domain/bus/device does not match",
        ),
        (
            "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
            None,
            33,
            0,
            "PCI domain/bus/device is unavailable/invalid",
        ),
        (
            "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
            0,
            33,
            True,
            "PCI domain/bus/device is unavailable/invalid",
        ),
        (
            "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
            True,
            33,
            0,
            "PCI domain/bus/device is unavailable/invalid",
        ),
        (
            "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
            0x10000,
            33,
            0,
            "PCI domain/bus/device does not match",
        ),
        (
            "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
            0,
            256,
            0,
            "PCI domain/bus/device does not match",
        ),
        (
            "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
            0,
            33,
            32,
            "PCI domain/bus/device does not match",
        ),
        (
            "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
            0,
            "00000000:22:00.0",
            0,
            "PCI bus does not match",
        ),
        (
            "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
            0,
            None,
            0,
            "full PCI bus ID must be a string",
        ),
    ],
)
def test_gpu_validation_rejects_uuid_or_pci_mismatch(
    observed_uuid: str,
    observed_domain: object,
    observed_pci: object,
    observed_device: object,
    error: str,
) -> None:
    identity = {
        "physical_gpu_index": 0,
        "uuid": "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
        "pci_bus_id": "00000000:21:00.0",
    }
    properties = SimpleNamespace(
        uuid=observed_uuid,
        pci_domain_id=observed_domain,
        pci_bus_id=observed_pci,
        pci_device_id=observed_device,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: properties,
        )
    )
    with pytest.raises(subject.LooK5ScoreError, match=error):
        subject._verify_visible_cuda_device(fake_torch, identity)


@pytest.mark.parametrize(
    ("pci_bus_id", "error"),
    [
        ("21:00.0", "PCI bus ID is unavailable/invalid"),
        ("00000000:21:20.0", "component is out of range"),
        (
            "00000000:21:00.1",
            "PCI function is not independently verifiable",
        ),
    ],
)
def test_gpu_validation_rejects_unverifiable_physical_bdf(
    pci_bus_id: str, error: str
) -> None:
    identity = {
        "physical_gpu_index": 0,
        "uuid": "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
        "pci_bus_id": pci_bus_id,
    }
    properties = SimpleNamespace(
        uuid=identity["uuid"],
        pci_domain_id=0,
        pci_bus_id=33,
        pci_device_id=0,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: properties,
        )
    )
    with pytest.raises(subject.LooK5ScoreError, match=error):
        subject._verify_visible_cuda_device(fake_torch, identity)


@pytest.mark.parametrize(
    "value",
    [
        None,
        b"short",
        "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c-extra",
        "xxba3a7f4c74c84922364d031c5187624cyy",
        "GPU-ba3a7f4c74c8-4922-364d-031c5187624c",
    ],
)
def test_gpu_uuid_normalisation_rejects_missing_or_malformed(value: object) -> None:
    with pytest.raises(subject.LooK5ScoreError, match="UUID is unavailable/invalid"):
        subject._normalise_gpu_uuid(value)


def test_gpu_uuid_normalisation_accepts_only_explicit_supported_forms() -> None:
    expected = "ba3a7f4c74c84922364d031c5187624c"
    assert subject._normalise_gpu_uuid(bytes.fromhex(expected)) == expected
    assert subject._normalise_gpu_uuid(expected.upper()) == expected
    assert (
        subject._normalise_gpu_uuid(
            "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c"
        )
        == expected
    )


def test_foreign_gpu_guard_allows_self_tree_and_rejects_foreign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject.gpu_runtime, "_descendant_pids", lambda: {10, 11})
    monkeypatch.setattr(
        subject.gpu_runtime,
        "_external_gpu_processes",
        lambda index: [{"pid": 10}, {"pid": 11}],
    )
    subject._assert_no_foreign_gpu_processes(0)
    monkeypatch.setattr(
        subject.gpu_runtime,
        "_external_gpu_processes",
        lambda index: [{"pid": 10}, {"pid": 99}],
    )
    with pytest.raises(subject.LooK5ScoreError, match="foreign GPU process"):
        subject._assert_no_foreign_gpu_processes(0)


def test_finalize_rejects_draw_or_seed_drift_before_loading_metric_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "score"
    output.mkdir()
    plan = {
        "bootstrap_contract": {
            "draws": subject.BOOTSTRAP_DRAWS,
            "seed": subject.BOOTSTRAP_SEED,
        }
    }
    monkeypatch.setattr(
        subject,
        "load_score_plan",
        lambda path, sha: (plan, {"path": str(path), "sha256": sha}),
    )
    with pytest.raises(subject.LooK5ScoreError, match="draws/seed"):
        subject.finalize_score(
            plan_path=tmp_path / "plan.json",
            plan_sha256="a" * 64,
            bert_manifest_path=tmp_path / "bert.json",
            bert_manifest_sha256="b" * 64,
            mpnet_manifest_path=tmp_path / "mpnet.json",
            mpnet_manifest_sha256="c" * 64,
            output_dir=output,
            draws=9999,
            seed=subject.BOOTSTRAP_SEED,
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda plan: plan["bootstrap_contract"].__setitem__("seed", 7),
        lambda plan: plan["k_adequacy_rule"].__setitem__(
            "k5_minus_k4_point_drift_threshold", 0.006
        ),
        lambda plan: plan["k_adequacy_rule"].__setitem__(
            "increase_k_if", ["drifted"]
        ),
        lambda plan: plan["k_adequacy_rule"].__setitem__(
            "stochastic_prefixes", "historical greedy accidentally included"
        ),
    ],
)
def test_load_plan_rejects_bootstrap_threshold_and_k_rule_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator,
) -> None:
    monkeypatch.setattr(subject, "_assert_binding", lambda *args, **kwargs: Path("/x"))
    monkeypatch.setattr(subject, "_assert_source_bundle", lambda sources: {})
    plan = _minimal_plan()
    mutator(plan)
    path = tmp_path / "plan.json"
    _sealed, sha = _write_sealed(path, plan)
    with pytest.raises(subject.LooK5ScoreError, match="bootstrap/K-adequacy"):
        subject.load_score_plan(path, sha)


def test_load_plan_accepts_exact_frozen_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject, "_assert_binding", lambda *args, **kwargs: Path("/x"))
    monkeypatch.setattr(subject, "_assert_source_bundle", lambda sources: {})
    path = tmp_path / "plan.json"
    sealed, sha = _write_sealed(path, _minimal_plan())
    observed, binding = subject.load_score_plan(path, sha)
    assert observed == sealed
    assert binding["payload_sha256"] == sealed["integrity"]["payload_sha256"]


def test_v2_scorer_rejects_sealed_v1_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject, "_assert_binding", lambda *args, **kwargs: Path("/x"))
    monkeypatch.setattr(subject, "_assert_source_bundle", lambda sources: {})
    plan = _minimal_plan()
    plan["schema_version"] = "chk3-core8-loo-vllm-k5-score-plan-v1"
    plan["evaluation_id"] = (
        f"{subject.EVALUATION_ID}-raw-semantic-b10000-v1"
    )
    path = tmp_path / "v1-plan.json"
    _sealed, sha = _write_sealed(path, plan)
    with pytest.raises(subject.LooK5ScoreError, match="score plan contract drift"):
        subject.load_score_plan(path, sha)


def test_source_bundle_rejects_any_bound_implementation_drift(tmp_path: Path) -> None:
    sources = {}
    for role in subject.SOURCE_ROLES:
        path = tmp_path / f"{role}.py"
        path.write_text(f"# {role}\n", encoding="utf-8")
        sources[role] = subject._file_binding(path)
    assert set(subject._assert_source_bundle(sources)) == set(subject.SOURCE_ROLES)
    drifted = Path(sources["semantic_long_text_engine"]["path"])
    drifted.write_text("# changed after plan\n", encoding="utf-8")
    with pytest.raises(subject.LooK5ScoreError, match="binding drift"):
        subject._assert_source_bundle(sources)


def _metric_bundle_fixture(
    tmp_path: Path, *, metric: str = "mpnet"
) -> tuple[dict, dict, Path, str]:
    sources = {}
    for role in subject.SOURCE_ROLES:
        source_path = tmp_path / f"source-{role}.py"
        source_path.write_text(f"# {role}\n", encoding="utf-8")
        sources[role] = subject._file_binding(source_path)
    plan_binding = {
        "path": str(tmp_path / "plan.json"),
        "sha256": "a" * 64,
        "bytes": 1,
        "payload_sha256": "b" * 64,
    }
    canonical_binding = {
        "path": str(tmp_path / "generation.jsonl"),
        "sha256": "c" * 64,
        "bytes": 1,
        "rows": 1,
    }
    semantic_binding = {
        "path": str(tmp_path / "semantic.json"),
        "sha256": "d" * 64,
        "bytes": 1,
        "payload_sha256": "e" * 64,
    }
    plan = {
        "inputs": {
            "canonical_generations": canonical_binding,
            "semantic_manifest": semantic_binding,
        },
        "sources": sources,
    }
    row_path = tmp_path / f"{metric}-rows.jsonl"
    row_path.write_text('{"row":1}\n', encoding="utf-8")
    manifest_payload = {
        "schema_version": subject.METRIC_MANIFEST_SCHEMA,
        "status": "complete",
        "evaluation_id": subject.SCORE_EVALUATION_ID,
        "metric_worker": metric,
        "inputs": {
            "score_plan": plan_binding,
            "canonical_generations": canonical_binding,
            "semantic_manifest": semantic_binding,
        },
        "coverage": {
            "rows": 1,
            "raw_all_rows_scored": True,
            "gate_filtered_rows": 0,
            "gate_zero_penalized_rows": 0,
            "empty_answer_rows": 0,
            "empty_answer_all_raw_metrics_exact_zero": True,
        },
        "sources": sources,
        "artifacts": {"row_scores": subject._file_binding(row_path, rows=1)},
    }
    manifest_path = tmp_path / f"{metric}-manifest.json"
    _sealed, manifest_sha = _write_sealed(manifest_path, manifest_payload)
    return plan, plan_binding, manifest_path, manifest_sha


def test_metric_bundle_rejects_input_row_hash_and_seal_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject, "EXPECTED_ROWS", 1)
    monkeypatch.setattr(subject, "EXPECTED_EMPTY_ANSWER_ROWS", 0)
    plan, plan_binding, manifest_path, manifest_sha = _metric_bundle_fixture(tmp_path)
    _manifest, rows, _binding = subject.load_metric_bundle(
        metric="mpnet",
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        plan=plan,
        plan_binding=plan_binding,
    )
    assert rows == [{"row": 1}]

    row_path = Path(
        json.loads(manifest_path.read_text(encoding="utf-8"))["artifacts"][
            "row_scores"
        ]["path"]
    )
    row_path.write_text('{"row":2}\n', encoding="utf-8")
    with pytest.raises(subject.LooK5ScoreError, match="binding drift"):
        subject.load_metric_bundle(
            metric="mpnet",
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
            plan=plan,
            plan_binding=plan_binding,
        )

    other = tmp_path / "unsealed-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["status"] = "tampered"
    other.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(subject.LooK5ScoreError, match="integrity"):
        subject.load_metric_bundle(
            metric="mpnet",
            manifest_path=other,
            manifest_sha256=subject._sha256_file(other),
            plan=plan,
            plan_binding=plan_binding,
        )


def test_metric_bundle_rejects_plan_input_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject, "EXPECTED_ROWS", 1)
    monkeypatch.setattr(subject, "EXPECTED_EMPTY_ANSWER_ROWS", 0)
    plan, plan_binding, manifest_path, manifest_sha = _metric_bundle_fixture(tmp_path)
    drifted = copy.deepcopy(plan)
    drifted["inputs"]["semantic_manifest"] = {
        **drifted["inputs"]["semantic_manifest"],
        "sha256": "f" * 64,
    }
    with pytest.raises(subject.LooK5ScoreError, match="contract drift"):
        subject.load_metric_bundle(
            metric="mpnet",
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
            plan=drifted,
            plan_binding=plan_binding,
        )
