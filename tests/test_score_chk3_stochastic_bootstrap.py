from __future__ import annotations

import copy
from pathlib import Path

import pytest

from jobs.eval import score_chk3_stochastic_bootstrap as subject
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


class FakeBERT:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    def score(self, candidates, references):
        self.calls.append((list(candidates), list(references)))
        size = len(candidates)
        return {
            "bertscore_precision": [0.7] * size,
            "bertscore_recall": [0.9] * size,
            "bertscore_f1": [0.8] * size,
        }

    def semantic_metadata(self):
        return {"chunk_audit": {"silent_truncation": False}}


class FakeMPNet:
    def score(self, candidates, references):
        return [0.6] * len(candidates)

    def semantic_metadata(self):
        return {"chunk_audit": {"silent_truncation": False}}


def _generation_row(
    *, model_id: str, sample_id: str, meeting_id: str, replicate_id: int, identity: bool
) -> dict:
    source = f"In January 2024, the reported rate was {replicate_id + 1}.0 percent."
    answer = source
    return {
        "model_id": model_id,
        "model_label": model_id,
        "sample_id": sample_id,
        "meeting_id": meeting_id,
        "replicate_id": replicate_id,
        "replicate_seed": subject.REPLICATE_SEEDS[replicate_id],
        "seed": 100 + replicate_id,
        "normalized_identity": identity,
        "source_analysis": source,
        "answer": answer,
        "reference_minutes": source,
        "generated_text": f"Reasoning.\n</think>\n{answer}",
        "hit_eos": True,
        "cap_reached": False,
        "think_boundary_count": 1,
        "exact_boundary_delimiter_count": 1,
        "final_answer_single_paragraph": True,
        "full_token_4gram_repetition": 0.0,
        "tail_token_4gram_repetition": 0.0,
        "strict_periodic_tail": False,
        "quality_valid": True,
        "quality_failures": [],
        "input_truncated": False,
        "completion_sha256": f"{replicate_id + 1:064x}",
        "answer_sha256": f"{replicate_id + 11:064x}",
        "reference_minutes_sha256": f"{replicate_id + 21:064x}",
        "generated_token_ids_sha256": f"{replicate_id + 31:064x}",
        "finish_reason": "eos_token",
        "generation_metrics": {
            metric: True for metric in subject.GENERATION_METRICS
        },
    }


def _small_runs() -> dict[str, dict]:
    samples = (
        ("sample-0", "m0", True),
        ("sample-1", "m0", False),
        ("sample-2", "m1", False),
        ("sample-3", "m1", False),
    )
    runs = {}
    for model_id in subject.MODEL_ORDER:
        rows = [
            _generation_row(
                model_id=model_id,
                sample_id=sample_id,
                meeting_id=meeting_id,
                replicate_id=replicate_id,
                identity=identity,
            )
            for sample_id, meeting_id, identity in samples
            for replicate_id in subject.REPLICATE_IDS
        ]
        runs[model_id] = {
            "results": rows,
            "generation_contract": {"do_sample": True},
            "manifest": {
                "sample_manifest": {"sha256": "a" * 64},
                "source_artifact_sha256s": {"runner": "b" * 64},
            },
        }
    return runs


def _sample_records() -> list[dict]:
    return [
        {"sample_id": "sample-0", "meeting_id": "m0", "normalized_identity": True},
        {"sample_id": "sample-1", "meeting_id": "m0", "normalized_identity": False},
        {"sample_id": "sample-2", "meeting_id": "m1", "normalized_identity": False},
        {"sample_id": "sample-3", "meeting_id": "m1", "normalized_identity": False},
    ]


def _score_rows(value_by_model=None) -> list[dict]:
    values = value_by_model or {model_id: 1.0 for model_id in subject.MODEL_ORDER}
    rows = []
    for model_id in subject.MODEL_ORDER:
        for sample in _sample_records():
            for replicate_id in subject.REPLICATE_IDS:
                identity = sample["normalized_identity"]
                metric_value = float(values[model_id])
                rows.append(
                    {
                        "model_id": model_id,
                        "sample_id": sample["sample_id"],
                        "meeting_id": sample["meeting_id"],
                        "replicate_id": replicate_id,
                        "normalized_identity": identity,
                        "generation_metrics": {
                            metric: bool(metric_value)
                            for metric in subject.GENERATION_METRICS
                        },
                        "semantic_status": (
                            "excluded_normalized_identity" if identity else "scored"
                        ),
                        "six_metrics": {
                            **{
                                metric: metric_value
                                for metric in subject.GENERATION_METRICS
                            },
                            **{
                                metric: None if identity else metric_value
                                for metric in subject.SEMANTIC_METRICS
                            },
                        },
                    }
                )
    return rows


def _small_view() -> dict:
    return {
        "view_id": "synthetic",
        "sample_ids": [row["sample_id"] for row in _sample_records()],
        "generation_prompts": 4,
        "semantic_prompts": 3,
        "inferential_conclusion_authorized": True,
    }


def test_identity_is_excluded_and_hard_gate_failure_gets_both_semantic_zeros() -> None:
    runs = _small_runs()
    failed = runs["chk3"]["results"][6]
    failed["answer"] = failed["answer"].replace("2.0", "9.0")
    failed["generated_text"] = f"Reasoning.\n</think>\n{failed['answer']}"
    failed["generation_metrics"]["numeric_fidelity"] = False
    bert = FakeBERT()

    rows, failures, audit = subject.build_scored_rows(
        runs=runs, bert=bert, mpnet=FakeMPNet(), formal=False
    )

    # 60 total rows minus 15 identity rows minus one failed non-identity row.
    assert len(bert.calls) == 1
    assert len(bert.calls[0][0]) == 44
    identity = next(
        row
        for row in rows
        if row["model_id"] == "chk0"
        and row["sample_id"] == "sample-0"
        and row["replicate_id"] == 0
    )
    assert identity["six_metrics"]["mpnet_cosine"] is None
    assert identity["semantic_status"] == "excluded_normalized_identity"
    penalized = next(
        row
        for row in rows
        if row["model_id"] == "chk3"
        and row["sample_id"] == "sample-1"
        and row["replicate_id"] == 1
    )
    assert penalized["six_metrics"]["mpnet_cosine"] == 0.0
    assert penalized["six_metrics"]["bertscore_f1"] == 0.0
    assert penalized["semantic_zero_penalty_applied"] is True
    assert failures[0]["failed_generation_metrics"] == ["numeric_fidelity"]
    assert audit["hard_gate_failure_policy"] == "both_semantic_metrics_fixed_to_zero"


@pytest.mark.parametrize("drift", ("manifest_source", "row_prompt", "row_seed"))
def test_cross_model_pairing_rejects_source_prompt_or_seed_drift(drift: str) -> None:
    runs = _small_runs()
    if drift == "manifest_source":
        runs["chk3"]["manifest"]["source_artifact_sha256s"]["runner"] = "c" * 64
    elif drift == "row_prompt":
        runs["chk3"]["results"][0]["source_prompt_sha256"] = "d" * 64
    else:
        runs["chk3"]["results"][0]["row_seed"] = 999
    with pytest.raises(subject.StochasticBootstrapError, match="drift"):
        subject.validate_run_matrix(runs, formal=False)


def test_formal_selection_and_greedy_anchors_are_exactly_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    selection = (
        root
        / "docs/summary/20260811T003000Z/chk3_native_analysis_to_minutes_cp250_eval/samples_n12.json"
    )
    selection_binding = subject._file_binding(selection, sealed=True)
    assert selection_binding["sha256"] == subject.SELECTION_N12_FILE_SHA256
    assert (
        selection_binding["payload_sha256"]
        == subject.SELECTION_N12_PAYLOAD_SHA256
    )
    greedy = (
        root
        / "output/evaluation/main/chk3_checkpoint_selection_cp318_20260811_v1/six_metrics/chk0_chk1_chk3_cp318_n12_cpu_v1.json"
    )
    payload, binding = subject._load_greedy_anchor(
        greedy, "acaea49f4865f080623bf42c2397abe4b18fc84411e7800463e08bbc0ae6b84a"
    )
    assert binding["payload_sha256"] == subject.GREEDY_ANCHOR_PAYLOAD_SHA256
    assert payload["selected_checkpoint"]["checkpoint_step"] == 318
    semantic = subject._file_binding(
        root / "configs/main/checkpoint_eval_semantic_models.json", sealed=True
    )
    assert semantic["sha256"] == subject.SEMANTIC_MANIFEST_FILE_SHA256
    assert semantic["payload_sha256"] == subject.SEMANTIC_MANIFEST_PAYLOAD_SHA256


def test_formal_view_denominators_are_exact_and_only_row_disjoint_is_primary() -> None:
    # First nine meetings contain 131 rows; the last four contain 59.
    meeting_sizes = [15, 15, 15, 15, 15, 14, 14, 14, 14, 15, 15, 15, 14]
    samples = []
    for meeting_index, size in enumerate(meeting_sizes):
        for row_index in range(size):
            sample_id = f"sample-{meeting_index:02d}-{row_index:02d}"
            # Twelve identities in the selected-meeting side, seven strict-side.
            identity = (
                meeting_index < 9 and len([s for s in samples if s["normalized_identity"]]) < 12
            ) or (meeting_index >= 9 and row_index == 0) or (
                meeting_index in (9, 10, 11) and row_index == 1
            )
            samples.append(
                {
                    "sample_id": sample_id,
                    "meeting_id": f"m{meeting_index:02d}",
                    "normalized_identity": identity,
                }
            )
    # Exactly one identity plus 11 non-identities across all first nine meetings.
    selected = []
    for meeting_index in range(9):
        candidates = [
            sample for sample in samples if sample["meeting_id"] == f"m{meeting_index:02d}"
        ]
        if meeting_index == 0:
            selected.append(next(sample for sample in candidates if sample["normalized_identity"]))
        else:
            selected.append(next(sample for sample in candidates if not sample["normalized_identity"]))
    remaining_nonidentity = [
        sample
        for sample in samples
        if sample["meeting_id"].startswith("m0")
        and not sample["normalized_identity"]
        and sample not in selected
    ][:3]
    selected.extend(remaining_nonidentity)
    assert len(samples) == 190
    assert sum(sample["normalized_identity"] for sample in samples) == 19
    assert len(selected) == 12
    assert sum(sample["normalized_identity"] for sample in selected) == 1

    views = subject.build_views(
        full_sample_manifest={"samples": samples},
        selection_sample_manifest={"samples": selected},
        formal=True,
    )

    assert {
        view_id: (
            view["generation_prompts"], view["semantic_prompts"], view["meetings"]
        )
        for view_id, view in views.items()
    } == {
        "full": (190, 171, 13),
        "row_disjoint": (178, 160, 13),
        "strict_meeting_disjoint": (59, 52, 4),
    }
    assert views["full"]["inferential_conclusion_authorized"] is False
    assert views["row_disjoint"]["inferential_conclusion_authorized"] is True
    assert views["strict_meeting_disjoint"]["inferential_conclusion_authorized"] is False
    assert views["strict_meeting_disjoint"]["excluded_prompts"] == 131
    assert views["strict_meeting_disjoint"]["selection_anchor_rows"] == 12


def test_all_one_metrics_have_exact_one_ci_and_reproducible_shared_draw_hash() -> None:
    result, draws = subject.bootstrap_view(
        view=_small_view(),
        full_sample_records=_sample_records(),
        row_scores=_score_rows(),
        draws=40,
        seed=123,
    )
    repeated, repeated_draws = subject.bootstrap_view(
        view=_small_view(),
        full_sample_records=_sample_records(),
        row_scores=_score_rows(),
        draws=40,
        seed=123,
    )

    assert result["bootstrap_index_plan"]["sha256"] == repeated[
        "bootstrap_index_plan"
    ]["sha256"]
    assert draws == repeated_draws
    for model_id in subject.MODEL_ORDER:
        for metric in subject.SIX_METRICS:
            estimate = result["estimates"][model_id][metric]
            assert estimate["point_estimate"] == 1.0
            assert estimate["ci_lower"] == 1.0
            assert estimate["ci_upper"] == 1.0
            assert estimate["failed_draws"] == 0


def test_formal_2000_draw_index_plan_is_byte_reproducible() -> None:
    first = subject.make_bootstrap_plan(
        view_id="formal-repro",
        meeting_ids=("m0", "m1"),
        sample_ids=("s0", "s1", "s2"),
        draws=subject.BOOTSTRAP_DRAWS,
        seed=subject.BOOTSTRAP_SEED,
    )
    second = subject.make_bootstrap_plan(
        view_id="formal-repro",
        meeting_ids=("m0", "m1"),
        sample_ids=("s0", "s1", "s2"),
        draws=subject.BOOTSTRAP_DRAWS,
        seed=subject.BOOTSTRAP_SEED,
    )
    assert first.meeting_indices.shape == (2000, 2)
    assert first.replicate_indices.shape == (2000, 2, 3, 5)
    assert first.sha256 == second.sha256
    assert first.meeting_indices.tobytes() == second.meeting_indices.tobytes()
    assert first.replicate_indices.tobytes() == second.replicate_indices.tobytes()


def test_bootstrap_view_uses_runtime_replicate_profile(monkeypatch) -> None:
    """A K=10 profile must not inherit make_bootstrap_plan's K=5 default."""

    monkeypatch.setattr(subject, "REPLICATE_IDS", tuple(range(10)))
    records = [
        {"sample_id": "s0", "meeting_id": "m0", "normalized_identity": False}
    ]
    rows = []
    for model_id in subject.MODEL_ORDER:
        for replicate_id in subject.REPLICATE_IDS:
            metrics = {metric: 1.0 for metric in subject.SIX_METRICS}
            rows.append(
                {
                    "model_id": model_id,
                    "sample_id": "s0",
                    "replicate_id": replicate_id,
                    "generation_metrics": {
                        metric: True for metric in subject.GENERATION_METRICS
                    },
                    "six_metrics": metrics,
                }
            )

    result, _ = subject.bootstrap_view(
        view={
            "view_id": "k10",
            "sample_ids": ["s0"],
            "generation_prompts": 1,
            "semantic_prompts": 1,
            "inferential_conclusion_authorized": True,
        },
        full_sample_records=records,
        row_scores=rows,
        draws=4,
        seed=7,
    )

    plan = result["bootstrap_index_plan"]
    assert plan["replicates"] == 10
    assert plan["replicate_index_shape"] == [4, 1, 1, 10]


def test_shared_indices_preserve_constant_paired_semantic_difference() -> None:
    rows = _score_rows({"chk0": 0.2, "chk1": 0.2, "chk3": 0.4})
    # Generation metrics are contractual binaries; keep only semantic values fractional.
    for row in rows:
        for metric in subject.GENERATION_METRICS:
            row["generation_metrics"][metric] = True
            row["six_metrics"][metric] = 1.0
    result, _ = subject.bootstrap_view(
        view=_small_view(),
        full_sample_records=_sample_records(),
        row_scores=rows,
        draws=50,
        seed=999,
    )

    contrast = next(
        row
        for row in result["contrasts"]
        if row["contrast_id"] == "chk3_minus_chk1"
        and row["metric"] == "mpnet_cosine"
    )
    assert contrast["point_difference"] == pytest.approx(0.2)
    assert contrast["ci_lower"] == pytest.approx(0.2)
    assert contrast["ci_upper"] == pytest.approx(0.2)


def test_point_estimate_is_meeting_equal_not_row_equal() -> None:
    records = [
        {"sample_id": "a", "meeting_id": "m0", "normalized_identity": False},
        *[
            {"sample_id": f"b{i}", "meeting_id": "m1", "normalized_identity": False}
            for i in range(3)
        ],
    ]
    rows = []
    for model_id in subject.MODEL_ORDER:
        for sample in records:
            value = 0.0 if sample["meeting_id"] == "m0" else 1.0
            for replicate_id in subject.REPLICATE_IDS:
                rows.append(
                    {
                        "model_id": model_id,
                        "sample_id": sample["sample_id"],
                        "meeting_id": sample["meeting_id"],
                        "replicate_id": replicate_id,
                        "normalized_identity": False,
                        "generation_metrics": {
                            metric: bool(value)
                            for metric in subject.GENERATION_METRICS
                        },
                        "semantic_status": "scored",
                        "six_metrics": {metric: value for metric in subject.SIX_METRICS},
                    }
                )
    result, _ = subject.bootstrap_view(
        view={
            "view_id": "meeting-equal",
            "sample_ids": [sample["sample_id"] for sample in records],
            "generation_prompts": 4,
            "semantic_prompts": 4,
            "inferential_conclusion_authorized": True,
        },
        full_sample_records=records,
        row_scores=rows,
        draws=20,
        seed=5,
    )
    assert result["estimates"]["chk0"]["structure_delivery"][
        "point_estimate"
    ] == pytest.approx(0.5)


def test_per_metric_failure_counts_are_reported_at_row_prompt_and_meeting_levels() -> None:
    rows = _score_rows()
    failed = next(
        row
        for row in rows
        if row["model_id"] == "chk3"
        and row["sample_id"] == "sample-2"
        and row["replicate_id"] == 0
    )
    failed["generation_metrics"]["numeric_fidelity"] = False
    failed["six_metrics"]["numeric_fidelity"] = 0.0
    failed["semantic_status"] = "hard_gate_zero_penalty"
    failed["six_metrics"]["mpnet_cosine"] = 0.0
    failed["six_metrics"]["bertscore_f1"] = 0.0

    result, _ = subject.bootstrap_view(
        view=_small_view(),
        full_sample_records=_sample_records(),
        row_scores=rows,
        draws=20,
        seed=11,
    )

    numeric = result["estimates"]["chk3"]["numeric_fidelity"]
    assert numeric["observations"] == 20
    assert numeric["zero_or_fail_generation_count"] == 1
    assert numeric["prompts_with_any_failure"] == 1
    assert numeric["meetings_with_any_failure"] == 1
    semantic = result["estimates"]["chk3"]["mpnet_cosine"]
    assert semantic["observations"] == 15
    assert semantic["hard_gate_zero_penalty_generation_count"] == 1
    assert semantic["semantic_scored_generation_count"] == 14
    assert semantic["normalized_identity_excluded_generation_count"] == 5


def test_holm_and_exact_sign_flip_contracts() -> None:
    assert subject.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx(
        [0.03, 0.06, 0.06]
    )
    p_value, combinations = subject.exact_sign_flip_p_value([0.0] * 13)
    assert p_value == 1.0
    assert combinations == 8192
    with pytest.raises(subject.StochasticBootstrapError, match="1..13"):
        subject.exact_sign_flip_p_value([0.1] * 14)


def test_large_meeting_sign_flip_is_reproducible_monte_carlo() -> None:
    differences = [0.1 if index % 3 else -0.02 for index in range(128)]
    left = subject.paired_sign_flip_p_value(
        differences, monte_carlo_draws=2_000, seed=20260813
    )
    right = subject.paired_sign_flip_p_value(
        differences, monte_carlo_draws=2_000, seed=20260813
    )
    assert left == right
    p_value, assignments, method = left
    assert 0.0 < p_value <= 1.0
    assert assignments == 2_000
    assert method == "two_sided_monte_carlo_sign_flip_plus_one"


def test_sealed_bootstrap_result_reproducible() -> None:
    full = {"samples": _sample_records()}
    selection = {"samples": [copy.deepcopy(_sample_records()[3])]}
    result, draw_rows = subject.build_bootstrap_results(
        row_scores=_score_rows(),
        full_sample_manifest=full,
        selection_sample_manifest=selection,
        draws=10,
        seed=7,
        formal=False,
    )
    assert validate_manifest_integrity(result) == result["integrity"][
        "payload_sha256"
    ]
    assert len(draw_rows) == 30
    assert result["bootstrap_contract"]["shared_indices_across_models_and_metrics"]
    assert result["bootstrap_contract"]["weighted_composite_calculated"] is False
