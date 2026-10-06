from __future__ import annotations

from collections import Counter
import hashlib

from jobs.retrain_v2 import (
    evaluate_chk4_training_in_sample_stochastic_vllm_k10 as subject,
)


def test_frozen_training_population_is_unique_and_not_physical_schedule() -> None:
    rows = subject._training_samples()
    assert len(rows) == 211
    assert len({row["sample_id"] for row in rows}) == 211
    assert len({row["meeting_date"] for row in rows}) == 211
    assert Counter(row["direction"] for row in rows) == Counter(
        {"cut": 23, "hold": 156, "hike": 32}
    )
    assert Counter((row["direction"], row["magnitude_bp"]) for row in rows) == Counter(
        subject.EXPECTED_DIRECTION_MAGNITUDE_COUNTS
    )
    assert all(row["split"] == "train" for row in rows)
    assert all(row["prompt"].startswith("Make one policy decision") for row in rows)


def test_case_schedule_closes_and_pairs_all_four_models() -> None:
    samples = [
        {
            "sample_id": f"sample-{index:03d}",
            "panel": subject.PANEL,
        }
        for index in range(subject.MEETING_COUNT)
    ]
    models = [
        {"label": model["label"]}
        for model in subject.MODEL_SPECS
    ]
    cases = subject.base.canonical_cases(samples, models)
    assert len(cases) == 8_440
    assert Counter(row["assigned_shard"] for row in cases) == Counter({0: 4_220, 1: 4_220})
    blocks: dict[int, list[dict[str, object]]] = {}
    for row in cases:
        blocks.setdefault(int(row["paired_block_id"]), []).append(row)
    assert len(blocks) == 2_110
    assert all(len(block) == 4 for block in blocks.values())
    assert all(len({row["row_seed"] for row in block}) == 1 for block in blocks.values())
    assert all(len({row["assigned_shard"] for row in block}) == 1 for block in blocks.values())


def _fake_result(
    *, model_label: str, sample_id: str, replicate_id: int, exact: bool, invalid: bool
) -> dict[str, object]:
    return {
        "model_label": model_label,
        "sample_id": sample_id,
        "replicate_id": replicate_id,
        "decision_exact": exact,
        "decision_prediction": (
            None if invalid else {"direction": "hold", "magnitude_bp": 0}
        ),
    }


def test_exact_action_table_counts_invalid_outputs_as_errors(monkeypatch) -> None:
    monkeypatch.setattr(subject, "BOOTSTRAP_DRAWS", 100)
    samples = [
        {"sample_id": "cut", "direction": "cut", "magnitude_bp": 25},
        {"sample_id": "hold", "direction": "hold", "magnitude_bp": 0},
        {"sample_id": "hike", "direction": "hike", "magnitude_bp": 25},
    ]
    models = [
        {"label": "a", "display_name": "A"},
        {"label": "b", "display_name": "B"},
    ]
    results = []
    for model in models:
        for sample in samples:
            for replicate_id in range(10):
                invalid = model["label"] == "a" and sample["sample_id"] == "cut" and replicate_id == 0
                exact = not invalid and replicate_id < 5
                results.append(
                    _fake_result(
                        model_label=model["label"],
                        sample_id=sample["sample_id"],
                        replicate_id=replicate_id,
                        exact=exact,
                        invalid=invalid,
                    )
                )
    table = subject.exact_action_table(results, samples, models)
    all_actions = next(row for row in table if row["condition_id"] == "all")
    assert all_actions["support_meetings"] == 3
    assert all_actions["models"]["a"]["invalid_completions_counted_as_errors"] == 1
    assert all_actions["models"]["a"]["correct_completions"] == 14
    assert all_actions["models"]["a"]["total_completions"] == 30
    assert all_actions["models"]["b"]["correct_completions"] == 15


def test_report_language_is_explicitly_in_sample() -> None:
    rows = [
        {
            "policy_direction": "all",
            "magnitude_bp": None,
            "support_meetings": 211,
            "models": {
                model["label"]: {
                    "exact_action_rate": 0.5,
                    "correct_completions": 1055,
                    "total_completions": 2110,
                }
                for model in subject.MODEL_SPECS
            },
        }
    ]
    markdown = subject._markdown_table(rows)
    latex = subject._latex_table(rows)
    assert "in-sample diagnostic" in markdown
    assert "not external, held-out, temporal, or out-of-sample" in markdown
    assert "Post-Hoc Training-Meeting Exact-Action Fit" in latex
    assert "not evidence of held-out or temporal generalization" in latex


def test_worker_launcher_targets_training_profile_module() -> None:
    source = subject.Path(subject.__file__).read_text(encoding="utf-8")
    assert "jobs.retrain_v2.evaluate_chk4_training_in_sample_stochastic_vllm_k10" in source
    assert "models_processed_sequentially\": True" in source
    assert "physical_schedule_rows_admitted\": 0" in source


def test_chk0_runtime_allowlist_excludes_mutable_git_metadata() -> None:
    digest = hashlib.sha256()
    total = 0
    for path, (size, sha256) in sorted(subject.CHK0_RUNTIME_FILES.items()):
        digest.update(f"{path}\0{size}\0{sha256}\n".encode("utf-8"))
        total += size
    assert digest.hexdigest() == subject.CHK0_RUNTIME_PAYLOAD_SHA256
    assert total == 16_069_669_155
    assert not any(path.startswith(".git") for path in subject.CHK0_RUNTIME_FILES)


def test_profile_receipts_are_exactly_8440_and_not_external_compatibility() -> None:
    source_text = subject.Path(subject.__file__).read_text(encoding="utf-8")
    assert "formal_8440_row_training_meeting_diagnostic_generation_only" in source_text
    assert "prior_generation_rows_reused" in source_text
    assert "source_v4_generation_rows_reused" not in source_text
    assert "formal_930_row" not in source_text
    assert "base.smoke(" not in source_text
    assert "base.run(" not in source_text
    assert "base.authorize(" not in source_text


def test_manifest_and_report_bind_release_specific_caveats() -> None:
    source_text = subject.Path(subject.__file__).read_text(encoding="utf-8")
    assert '"canonical_dag_bindable": False' in source_text
    assert '"post2008_core": 102' in source_text
    assert '"pre2009_supplement": 109' in source_text
    assert "25 of 109 supplement meetings" in source_text
    assert "all 67 supplement hold labels" in source_text
    assert "Cut 100 bp (N=1)" in source_text
