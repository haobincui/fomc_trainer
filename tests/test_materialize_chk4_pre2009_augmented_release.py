from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pytest

from jobs.generation.generate_chk4_sft_targets import MANIFEST_SCHEMA
from jobs.retrain_v2 import materialize_chk4_pre2009_augmented_release as augmented
from open_r1.trainer.dataset_release import (
    CHK4_PRE2009_GRPO_ROLE,
    CHK4_PRE2009_SFT_ROLE,
    DatasetReleaseValidationError,
    verify_chk4_release_for_role,
)


ACTION_COUNTS = {
    "hold:0": 67,
    "hike:25": 20,
    "hike:50": 3,
    "cut:25": 9,
    "cut:50": 8,
    "cut:75": 2,
}


def test_summary_contract_path_is_pinned_to_grounding_v6(tmp_path: Path) -> None:
    assert (
        augmented.SUPPLEMENT_SUMMARY_PROMPT_CONTRACT_FILE
        == "prompt_contract.grounding-v6.json"
    )
    teacher_root = tmp_path / "teacher_targets/supplement"
    teacher_root.mkdir(parents=True)
    summary_contracts = [
        relative
        for relative in augmented._teacher_required_files(teacher_root)
        if relative.startswith("summaries/prompt_contract.")
    ]
    assert summary_contracts == [
        "summaries/prompt_contract.grounding-v6.json",
        "summaries/prompt_contract.grounding-v5.json",
    ]
    assert (
        augmented.SUPPLEMENT_BLIND_PROMPT_CONTRACT_FILE
        == augmented.SUPPLEMENT_TEACHER_PROMPT_CONTRACT_FILE
        == "prompt_contract.qualitative-v2.json"
    )
    required = augmented._teacher_required_files(teacher_root)
    assert "blind_predictions/prompt_contract.qualitative-v2.json" in required
    assert "teacher_targets/supplement/prompt_contract.qualitative-v2.json" in required


def _alpha(index: int) -> str:
    value = index
    output = ""
    while True:
        output = chr(ord("a") + value % 26) + output
        value = value // 26 - 1
        if value < 0:
            return output


def _source_row(index: int, direction: str, magnitude: int) -> dict[str, object]:
    sample_id = f"dec-{index:024x}"
    prompt = (
        'Make one policy decision from {"analysis":"Inflation, employment, and '
        f'financial conditions were mixed in synthetic case {_alpha(index)}."}}'
    )
    gold = augmented._gold_text(direction, magnitude)
    response = (
        "The qualitative balance of risks supports this stance." + "\n</think>\n" + gold
    )
    return {
        "schema_version": augmented.SOURCE_ROW_SCHEMA,
        "sample_id": sample_id,
        "meeting_date": (date(1990, 1, 1) + timedelta(days=index)).isoformat(),
        "population": "synthetic",
        "population_role": "core" if index % 2 == 0 else "supplement",
        "source_ids": [f"source-{index:04d}"],
        "direction": direction,
        "magnitude_bp": magnitude,
        "prompt": prompt,
        "response": response,
        "prompt_sha256": augmented._sha256_text(prompt),
        "response_sha256": augmented._sha256_text(response),
        "gold_sha256": augmented._sha256_text(gold),
    }


def _schedule_population() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 1
    for direction, magnitude, count in (
        ("hold", 0, 156),
        ("hike", 25, 32),
        ("cut", 25, 23),
    ):
        for _ in range(count):
            rows.append(_source_row(index, direction, magnitude))
            index += 1
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(augmented._canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _supplement_teacher(
    root: Path, *, action_counts: dict[str, int] | None = None
) -> Path:
    pipeline = root
    teacher = pipeline / "teacher_targets/supplement"
    counts = dict(ACTION_COUNTS if action_counts is None else action_counts)
    identity = ["deepseek-v4-pro", "fp-fixed"]

    def prompt_contract(
        stage: str, *, repair_attempts: int = 1
    ) -> tuple[dict[str, object], str]:
        payload: dict[str, object] = {
            "schema_version": augmented.SUPPLEMENT_PROVIDER_CONTRACT_SCHEMA,
            "stage": stage,
            "system_prompt": f"system prompt for {stage}",
            "repair_system_prompt": f"repair prompt for {stage}",
            "provider": {"model": "deepseek-v4-pro"},
            "model_fallback": "forbidden",
            "repair_attempts": repair_attempts,
            "code_sha256": "f" * 64,
        }
        digest = augmented._sha256_text(augmented._canonical_json(payload))
        return {**payload, "contract_sha256": digest}, digest

    summary_contract, summary_contract_sha = prompt_contract(
        "summaries", repair_attempts=2
    )
    prior_summary_contract = dict(summary_contract)
    prior_summary_contract.pop("contract_sha256")
    prior_summary_contract["repair_attempts"] = 1
    prior_summary_contract["code_sha256"] = "e" * 64
    prior_summary_contract_sha = augmented._sha256_text(
        augmented._canonical_json(prior_summary_contract)
    )
    prior_summary_contract["contract_sha256"] = prior_summary_contract_sha
    blind_contract, blind_contract_sha = prompt_contract(
        "blind_predictions", repair_attempts=2
    )
    target_contract, target_contract_sha = prompt_contract(
        "teacher_targets", repair_attempts=2
    )

    manifests: list[dict[str, object]] = []
    sft_rows: list[dict[str, object]] = []
    target_prepared: list[dict[str, object]] = []
    target_teacher_rows: list[dict[str, object]] = []
    admitted_rows: list[dict[str, object]] = []
    summary_requests: list[dict[str, object]] = []
    brief_rows: list[dict[str, object]] = []
    summary_teacher_rows: list[dict[str, object]] = []
    blind_teacher_rows: list[dict[str, object]] = []
    blind_predictions: list[dict[str, object]] = []
    index = 0
    for action, count in counts.items():
        direction, rendered_magnitude = action.split(":", 1)
        magnitude = int(rendered_magnitude)
        for _ in range(count):
            sample_id = f"dec-{(10_000 + index):024x}"
            meeting_date = (date(1993, 3, 23) + timedelta(days=index * 30)).isoformat()
            token = _alpha(index)
            brief = (
                "Inflation and labor conditions were mixed while financial "
                f"risks remained balanced in case {token}."
            )
            prompt = augmented.render_student_prompt(brief)
            summary_prompt = f"Summarize the supplied evidence for case {token}."
            summary_input_sha = augmented._sha256_text(f"evidence-{token}")
            gold = augmented._gold_text(direction, magnitude)
            reasoning = (
                "The qualitative evidence and balance of risks support this stance."
            )
            response = reasoning + "\n</think>\n" + gold
            provider = {
                "returned_model": identity[0],
                "system_fingerprint": identity[1],
                "finish_reason": "stop",
                "created": 1,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            source_id = f"canonical-loo-d1:synthetic:{meeting_date}:topic-{token}"
            admitted_rows.append(
                {
                    "schema_version": "chk4-decision-supplement-admission-v1",
                    "sample_id": sample_id,
                    "meeting_date": meeting_date,
                    "split": "train",
                    "population": augmented.SUPPLEMENT_POPULATION,
                    "population_role": "supplement",
                    "admission_profile": augmented.SUPPLEMENT_ADMISSION_PROFILE,
                    "status": "admitted",
                    "valid_atomic_topic_count": (
                        augmented.SUPPLEMENT_MIN_VALID_ATOMIC_TOPICS
                    ),
                    "category_coverage": sorted(
                        augmented.SUPPLEMENT_REQUIRED_CATEGORIES
                    ),
                    "source_ids": [source_id],
                    "source_provenance": [],
                    "gold": {"direction": direction, "magnitude_bp": magnitude},
                    "gold_sha256": augmented._sha256_text(gold),
                    "input_sha256": summary_input_sha,
                    "prompt_sha256": augmented._sha256_text(summary_prompt),
                    "prompt_tokens": 1,
                }
            )
            summary_requests.append(
                {
                    "schema_version": "chk4-decision-supplement-prepared-v1",
                    "sample_id": sample_id,
                    "prompt": summary_prompt,
                    "input_sha256": summary_input_sha,
                    "prompt_sha256": augmented._sha256_text(summary_prompt),
                }
            )
            brief_rows.append(
                {
                    "schema_version": augmented.SUPPLEMENT_SUMMARY_SCHEMA,
                    "sample_id": sample_id,
                    "meeting_decision_brief": brief,
                    "brief_sha256": augmented._sha256_text(brief),
                    "brief_tokens": 1,
                    "student_prompt_tokens": 1,
                    "input_sha256": summary_input_sha,
                    "generation_contract_sha256": summary_contract_sha,
                }
            )
            summary_provider = {
                **provider,
                "response_id": f"summary-{index}",
            }
            summary_content = json.dumps(
                {"meeting_decision_brief": brief}, separators=(",", ":")
            )
            summary_teacher_row: dict[str, object] = {
                "sample_id": sample_id,
                "attempt": (
                    "repair_2"
                    if index == augmented.SUPPLEMENT_EXPECTED_CANDIDATES - 1
                    else "primary"
                ),
                "provider": summary_provider,
                "reasoning_content": "native summary provenance",
                "content": summary_content,
                "revalidated_from": None,
            }
            if index < augmented.EXPECTED_SUMMARY_REVALIDATED_ROWS:
                prior_cache_key = augmented._summary_cache_key(
                    sample_id=sample_id,
                    input_sha256=summary_input_sha,
                    prompt_sha256=augmented._sha256_text(summary_prompt),
                    contract_sha256=prior_summary_contract_sha,
                )
                prior_cache_relative = (
                    "cache/summaries/accepted/"
                    f"{prior_cache_key[:2]}/{prior_cache_key}.json"
                )
                prior_cache = {
                    "schema_version": augmented.SUPPLEMENT_PROVIDER_CACHE_SCHEMA,
                    "status": "accepted",
                    "stage": "summaries",
                    "cache_key": prior_cache_key,
                    "sample_id": sample_id,
                    "input_sha256": summary_input_sha,
                    "prompt_sha256": augmented._sha256_text(summary_prompt),
                    "gold_sha256": None,
                    "contract_sha256": prior_summary_contract_sha,
                    "attempt": "primary",
                    "provider": summary_provider,
                    "provider_raw": {
                        "reasoning_content": "native summary provenance",
                        "content": summary_content,
                    },
                    "target": {
                        "meeting_decision_brief": brief,
                        "brief_sha256": augmented._sha256_text(brief),
                        "brief_tokens": 1,
                        "student_prompt_tokens": 1,
                    },
                }
                prior_cache_path = pipeline / prior_cache_relative
                _write_json(prior_cache_path, prior_cache)
                summary_teacher_row["revalidated_from"] = {
                    "path": prior_cache_relative,
                    "sha256": augmented._sha256_file(prior_cache_path),
                    "contract_path": (
                        "summaries/"
                        + augmented.SUPPLEMENT_PRIOR_SUMMARY_PROMPT_CONTRACT_FILE
                    ),
                    "contract_sha256": prior_summary_contract_sha,
                }
            summary_teacher_rows.append(summary_teacher_row)
            blind_teacher_rows.append(
                {
                    "sample_id": sample_id,
                    "attempt": "primary",
                    "provider": {**provider, "response_id": f"blind-{index}"},
                    "reasoning_content": "native blind provenance",
                    "content": json.dumps(
                        {
                            "reasoning": "Qualitative blind audit reasoning.",
                            "direction": direction,
                            "magnitude_bp": magnitude,
                        },
                        separators=(",", ":"),
                    ),
                }
            )
            blind_predictions.append(
                {
                    "schema_version": augmented.SUPPLEMENT_BLIND_SCHEMA,
                    "sample_id": sample_id,
                    "reasoning": "Qualitative blind audit reasoning.",
                    "direction": direction,
                    "magnitude_bp": magnitude,
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "analysis_sha256": augmented._sha256_text(brief),
                    "contract_sha256": blind_contract_sha,
                }
            )
            target_prepared.append(
                {
                    "schema_version": augmented.SUPPLEMENT_TARGET_SCHEMA,
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "input_sha256": augmented._sha256_text(brief),
                    "prompt_sha256": augmented._sha256_text(prompt),
                    "gold_sha256": augmented._sha256_text(gold),
                }
            )
            manifests.append(
                {
                    "schema_version": MANIFEST_SCHEMA,
                    "sample_id": sample_id,
                    "meeting_date": meeting_date,
                    "split": "train",
                    "population": augmented.SUPPLEMENT_POPULATION,
                    "population_role": "supplement",
                    "source_ids": [source_id],
                    "gold": {"direction": direction, "magnitude_bp": magnitude},
                    "input_sha256": augmented._sha256_text(brief),
                    "prompt_sha256": augmented._sha256_text(prompt),
                    "gold_sha256": augmented._sha256_text(gold),
                    "contract_sha256": target_contract_sha,
                }
            )
            sft_rows.append({"prompt": prompt, "response": response})
            target_teacher_rows.append(
                {
                    "sample_id": sample_id,
                    "attempt": "primary",
                    "provider": {**provider, "response_id": f"target-{index}"},
                    "reasoning_content": "native target provenance",
                    "content": json.dumps(
                        {
                            "reasoning": reasoning,
                            "direction": direction,
                            "magnitude_bp": magnitude,
                        },
                        separators=(",", ":"),
                    ),
                }
            )
            index += 1

    action_classes = sorted(counts)
    handoff_unsigned: dict[str, object] = {
        "schema_version": "chk1-source-handoff-v1",
        "source_plan": {"path": "synthetic", "sha256": "a" * 64},
        "registry_sha256": "b" * 64,
        "roster_sha256": "c" * 64,
        "ledgers": [
            {
                "population_id": "synthetic",
                "split": "train",
                "coverage_mode": "sparse",
                "meeting_count": len(manifests),
            }
        ],
    }
    handoff = {
        **handoff_unsigned,
        "payload_sha256": augmented._sha256_text(
            augmented._canonical_json(handoff_unsigned)
        ),
    }
    evidence = {
        "schema_version": "chk4-decision-supplement-evidence-audit-v2",
        "status": "complete",
        "admission_profile": augmented.SUPPLEMENT_ADMISSION_PROFILE,
        "candidate_count": augmented.SUPPLEMENT_EXPECTED_CANDIDATES,
        "admitted_count": len(manifests),
        "rejected_count": 0,
        "minimum_admitted": augmented.MIN_SUPPLEMENT_ROWS,
        "minimum_valid_atomic_topics": (augmented.SUPPLEMENT_MIN_VALID_ATOMIC_TOPICS),
        "required_categories": sorted(augmented.SUPPLEMENT_REQUIRED_CATEGORIES),
        "action_classes": action_classes,
        "source_handoff_payload_sha256": handoff["payload_sha256"],
        "forbidden_current_vintage_inputs": True,
        "meeting_identity_removed_from_model_input": True,
    }
    physical_counts = {
        "decision_sft": {"train": len(manifests), "validation": 13, "test": 13},
        "decision_grpo": {"train": len(manifests), "validation": 13, "test": 13},
    }
    training_summary = {
        "unique_counts": {
            "train": 102 + len(manifests),
            "validation": 13,
            "test": 13,
        },
        "physical_counts": physical_counts,
    }
    qa = {
        "schema_version": "chk4-decision-supplement-release-qa-v2",
        "status": "complete",
        "admission_profile": augmented.SUPPLEMENT_ADMISSION_PROFILE,
        "source_handoff_payload_sha256": handoff["payload_sha256"],
        "errors": [],
        "candidate_count": augmented.SUPPLEMENT_EXPECTED_CANDIDATES,
        "admitted_count": len(manifests),
        "rejected_count": 0,
        "action_classes": action_classes,
        "blind_metrics_are_audit_only": True,
        "combined_unique_counts": training_summary["unique_counts"],
        "combined_physical_counts": physical_counts,
        "provider_response_count": len(manifests) * 3,
        "provider_identity": identity,
    }

    _write_json(
        pipeline / "provider_identity.json",
        {
            "returned_model": identity[0],
            "system_fingerprint": identity[1],
        },
    )
    _write_jsonl(pipeline / "failures.jsonl", [])
    _write_jsonl(pipeline / "prepared/summary_requests.jsonl", summary_requests)
    _write_jsonl(pipeline / "manifests/admitted.jsonl", admitted_rows)
    _write_json(pipeline / "reports/evidence_audit.json", evidence)
    _write_json(
        pipeline / "reports/blind_prediction_metrics.json",
        {
            "status": "complete",
            "sample_count": len(manifests),
            "selection_policy": "audit_only_never_filter_training_rows",
        },
    )
    _write_json(pipeline / "sources/materialized/source_handoff.json", handoff)
    evidence_sha = augmented._sha256_file(pipeline / "reports/evidence_audit.json")
    qa["evidence_audit_sha256"] = evidence_sha
    _write_json(pipeline / "reports/release_qa.json", qa)
    for manifest in manifests:
        manifest["admission_profile"] = augmented.SUPPLEMENT_ADMISSION_PROFILE
        manifest["evidence_audit_sha256"] = evidence_sha
        manifest["source_handoff_payload_sha256"] = handoff["payload_sha256"]
    _write_json(
        pipeline / "summaries" / augmented.SUPPLEMENT_SUMMARY_PROMPT_CONTRACT_FILE,
        summary_contract,
    )
    _write_json(
        pipeline
        / "summaries"
        / augmented.SUPPLEMENT_PRIOR_SUMMARY_PROMPT_CONTRACT_FILE,
        prior_summary_contract,
    )
    _write_jsonl(pipeline / "summaries/failures.jsonl", [])
    _write_jsonl(pipeline / "summaries/meeting_decision_briefs.jsonl", brief_rows)
    _write_jsonl(pipeline / "summaries/teacher_responses.jsonl", summary_teacher_rows)
    _write_json(
        pipeline
        / "blind_predictions"
        / augmented.SUPPLEMENT_BLIND_PROMPT_CONTRACT_FILE,
        blind_contract,
    )
    _write_jsonl(pipeline / "blind_predictions/failures.jsonl", [])
    _write_jsonl(pipeline / "blind_predictions/predictions.jsonl", blind_predictions)
    _write_jsonl(
        pipeline / "blind_predictions/teacher_responses.jsonl", blind_teacher_rows
    )
    _write_json(pipeline / "training/combined_v1/summary.json", training_summary)

    def stage_summary(
        stage: str,
        contract_sha: str,
        *,
        prompt_contract_file: str = "prompt_contract.json",
        resumed_count: int = 0,
        revalidated_count: int = 0,
        api_requests: int | None = None,
    ) -> dict[str, object]:
        return {
            "schema_version": f"chk4-decision-supplement-{stage}-summary-v1",
            "status": "complete",
            "stage": stage,
            "prepared_count": len(manifests),
            "accepted_count": len(manifests),
            "failure_count": 0,
            "resumed_count": resumed_count,
            "revalidated_count": revalidated_count,
            "api_requests": len(manifests) if api_requests is None else api_requests,
            "contract_sha256": contract_sha,
            "prompt_contract_file": prompt_contract_file,
            "provider_identity": identity,
            "admission_profile": augmented.SUPPLEMENT_ADMISSION_PROFILE,
            "evidence_audit": {
                "path": str(pipeline / "reports/evidence_audit.json"),
                "sha256": evidence_sha,
                "schema_version": evidence["schema_version"],
            },
            "source_handoff": {
                "path": str(pipeline / "sources/materialized/source_handoff.json"),
                "sha256": augmented._sha256_file(
                    pipeline / "sources/materialized/source_handoff.json"
                ),
                "payload_sha256": handoff["payload_sha256"],
            },
            "admitted_manifest": {
                "path": str(pipeline / "manifests/admitted.jsonl"),
                "sha256": augmented._sha256_file(pipeline / "manifests/admitted.jsonl"),
                "rows": len(manifests),
            },
        }

    summary_stage = stage_summary(
        "summaries",
        summary_contract_sha,
        prompt_contract_file=augmented.SUPPLEMENT_SUMMARY_PROMPT_CONTRACT_FILE,
        resumed_count=augmented.EXPECTED_SUMMARY_REVALIDATED_ROWS,
        revalidated_count=augmented.EXPECTED_SUMMARY_REVALIDATED_ROWS,
        api_requests=augmented.EXPECTED_SUMMARY_PROVIDER_ROWS,
    )
    blind_stage = {
        **stage_summary(
            "blind_predictions",
            blind_contract_sha,
            prompt_contract_file=augmented.SUPPLEMENT_BLIND_PROMPT_CONTRACT_FILE,
        ),
        "execution_wrapper": {
            "path": str(augmented.SUPPLEMENT_BLIND_WRAPPER.resolve()),
            "sha256": augmented._sha256_file(augmented.SUPPLEMENT_BLIND_WRAPPER),
        },
        "reasoning_contract": augmented.SUPPLEMENT_QUALITATIVE_REASONING_CONTRACT,
    }
    target_stage = {
        **stage_summary(
            "teacher_targets",
            target_contract_sha,
            prompt_contract_file=augmented.SUPPLEMENT_TEACHER_PROMPT_CONTRACT_FILE,
        ),
        "teacher_output_mapping": {
            "reasoning": "json.loads(message.content)['reasoning']",
            "decision": "locally_serialized_canonical_gold",
            "native_reasoning_content": "provenance_only",
        },
        "execution_wrapper": {
            "path": str(augmented.SUPPLEMENT_TEACHER_WRAPPER.resolve()),
            "sha256": augmented._sha256_file(augmented.SUPPLEMENT_TEACHER_WRAPPER),
        },
        "reasoning_contract": augmented.SUPPLEMENT_QUALITATIVE_REASONING_CONTRACT,
    }
    _write_json(pipeline / "summaries/summary.json", summary_stage)
    _write_json(pipeline / "blind_predictions/summary.json", blind_stage)
    _write_json(
        pipeline / "summary.json",
        {
            "schema_version": "chk4-decision-supplement-pipeline-summary-v1",
            "population": augmented.SUPPLEMENT_POPULATION,
            "stages": {
                "evidence": evidence,
                "summaries": {
                    key: value
                    for key, value in summary_stage.items()
                    if key
                    not in {
                        "admission_profile",
                        "evidence_audit",
                        "source_handoff",
                        "admitted_manifest",
                    }
                },
                "blind_predictions": blind_stage,
                "teacher_targets": target_stage,
                "qa": qa,
            },
            "status": "complete",
        },
    )

    _write_json(
        teacher / augmented.SUPPLEMENT_TEACHER_PROMPT_CONTRACT_FILE,
        target_contract,
    )
    _write_jsonl(teacher / "failures.jsonl", [])
    _write_jsonl(teacher / "prepared/train.jsonl", target_prepared)
    _write_jsonl(teacher / "manifests/train.jsonl", manifests)
    _write_jsonl(teacher / "sft/train.jsonl", sft_rows)
    _write_jsonl(teacher / "teacher_responses/train.jsonl", target_teacher_rows)
    for split in ("validation", "test"):
        for family in ("prepared", "manifests", "sft", "teacher_responses"):
            _write_jsonl(teacher / family / f"{split}.jsonl", [])
    _write_json(teacher / "summary.json", target_stage)
    return teacher


def test_dynamic_schedule_is_complete_balanced_and_repeat_bounded() -> None:
    sources = _schedule_population()
    first, first_contract = augmented.build_schedule(sources)
    second, second_contract = augmented.build_schedule(copy.deepcopy(sources))

    assert first == second
    assert first_contract == second_contract
    assert len(first) == 312
    assert first_contract["optimizer_steps"] == 39
    assert first_contract["direction_counts"] == {
        "hold": 156,
        "hike": 78,
        "cut": 78,
    }
    assert first_contract["repeat_histograms"] == {
        "hold": {"1": 156},
        "hike": {"2": 18, "3": 14},
        "cut": {"3": 14, "4": 9},
    }
    assert first_contract["max_source_repeat"] == 4
    assert len({row["source_sample_id"] for row in first}) == 211
    for start in range(0, len(first), augmented.EFFECTIVE_BATCH_SIZE):
        window = first[start : start + augmented.EFFECTIVE_BATCH_SIZE]
        assert (
            Counter(row["direction"] for row in window) == augmented.PER_WINDOW_COUNTS
        )
        assert len({row["source_sample_id"] for row in window}) == 8


def test_supplement_requires_action_floors_and_train_only(tmp_path: Path) -> None:
    missing_rare = dict(ACTION_COUNTS)
    missing_rare["hike:50"] = 2
    missing_rare["hold:0"] = 68
    teacher = _supplement_teacher(tmp_path / "floor", action_counts=missing_rare)
    with pytest.raises(augmented.Pre2009ReleaseError, match="action floor failed"):
        augmented._verify_supplement_teacher(teacher)

    teacher = _supplement_teacher(tmp_path / "leak")
    train_manifests = [
        json.loads(line)
        for line in (teacher / "manifests/train.jsonl").read_text().splitlines()
    ]
    train_sft = [
        json.loads(line)
        for line in (teacher / "sft/train.jsonl").read_text().splitlines()
    ]
    manifest = train_manifests.pop(0)
    training = train_sft.pop(0)
    manifest["split"] = "validation"
    _write_jsonl(teacher / "manifests/train.jsonl", train_manifests)
    _write_jsonl(teacher / "sft/train.jsonl", train_sft)
    _write_jsonl(teacher / "manifests/validation.jsonl", [manifest])
    _write_jsonl(teacher / "sft/validation.jsonl", [training])
    with pytest.raises(augmented.Pre2009ReleaseError, match="leaked outside train"):
        augmented._verify_supplement_teacher(teacher)


def test_teacher_content_is_replayed_and_uses_producer_qualitative_predicate(
    tmp_path: Path,
) -> None:
    teacher = _supplement_teacher(tmp_path / "sft-drift")
    sft_path = teacher / "sft/train.jsonl"
    sft_rows = [json.loads(line) for line in sft_path.read_text().splitlines()]
    gold = sft_rows[0]["response"].split("\n</think>\n", 1)[1]
    sft_rows[0]["response"] = "Altered qualitative rationale.\n</think>\n" + gold
    _write_jsonl(sft_path, sft_rows)
    with pytest.raises(
        augmented.Pre2009ReleaseError, match="teacher-to-SFT reconstruction drift"
    ):
        augmented._verify_supplement_teacher(teacher)

    teacher = _supplement_teacher(tmp_path / "spelled-number")
    teacher_path = teacher / "teacher_responses/train.jsonl"
    teacher_rows = [json.loads(line) for line in teacher_path.read_text().splitlines()]
    content = json.loads(teacher_rows[0]["content"])
    content["reasoning"] = "One qualitative factor supports this stance."
    teacher_rows[0]["content"] = json.dumps(content, separators=(",", ":"))
    _write_jsonl(teacher_path, teacher_rows)
    sft_path = teacher / "sft/train.jsonl"
    sft_rows = [json.loads(line) for line in sft_path.read_text().splitlines()]
    sft_rows[0]["response"] = (
        content["reasoning"]
        + "\n</think>\n"
        + (sft_rows[0]["response"].split("\n</think>\n", 1)[1])
    )
    _write_jsonl(sft_path, sft_rows)
    assert augmented.qualitative_reasoning_has_number_or_date(content["reasoning"])
    with pytest.raises(
        augmented.Pre2009ReleaseError, match="teacher reasoning/decision drift"
    ):
        augmented._verify_supplement_teacher(teacher)


def test_summary_v6_revalidation_replays_v5_cache_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    teacher = _supplement_teacher(tmp_path / "summary-revalidation")
    pipeline = teacher.parents[1]
    verified = augmented._verify_supplement_teacher(teacher)
    assert len(verified["sources"]) == augmented.SUPPLEMENT_EXPECTED_CANDIDATES

    summary_rows = [
        json.loads(line)
        for line in (pipeline / "summaries/teacher_responses.jsonl")
        .read_text()
        .splitlines()
    ]
    assert (
        sum(row["revalidated_from"] is not None for row in summary_rows)
        == augmented.EXPECTED_SUMMARY_REVALIDATED_ROWS
    )
    assert summary_rows[-1]["attempt"] == "repair_2"

    current = json.loads(
        (
            pipeline / "summaries" / augmented.SUPPLEMENT_SUMMARY_PROMPT_CONTRACT_FILE
        ).read_text()
    )
    prior_path = (
        pipeline / "summaries" / augmented.SUPPLEMENT_PRIOR_SUMMARY_PROMPT_CONTRACT_FILE
    )
    prior = json.loads(prior_path.read_text())
    assert augmented._summary_contracts_are_revalidation_compatible(prior, current)
    incompatible = dict(prior)
    incompatible.pop("contract_sha256")
    incompatible["system_prompt"] = "incompatible prompt"
    incompatible["contract_sha256"] = augmented._sha256_text(
        augmented._canonical_json(incompatible)
    )
    assert not augmented._summary_contracts_are_revalidation_compatible(
        incompatible, current
    )

    provenance = summary_rows[0]["revalidated_from"]
    cache_path = pipeline / provenance["path"]
    cache = json.loads(cache_path.read_text())
    cache["target"] = {"meeting_decision_brief": "tampered"}
    _write_json(cache_path, cache)
    summary_rows[0]["revalidated_from"]["sha256"] = augmented._sha256_file(cache_path)
    _write_jsonl(pipeline / "summaries/teacher_responses.jsonl", summary_rows)
    with pytest.raises(
        augmented.Pre2009ReleaseError, match="legacy cache replay drift"
    ):
        augmented._verify_supplement_teacher(teacher)


def test_qualitative_v2_wrapper_and_blind_prediction_are_replayed(
    tmp_path: Path,
) -> None:
    teacher = _supplement_teacher(tmp_path / "blind-v2-wrapper-drift")
    pipeline = teacher.parents[1]
    blind_summary_path = pipeline / "blind_predictions/summary.json"
    blind_summary = json.loads(blind_summary_path.read_text())
    blind_summary["execution_wrapper"]["sha256"] = "0" * 64
    _write_json(blind_summary_path, blind_summary)
    with pytest.raises(
        augmented.Pre2009ReleaseError, match="execution wrapper SHA drift"
    ):
        augmented._verify_supplement_teacher(teacher)

    teacher = _supplement_teacher(tmp_path / "blind-v2-row-drift")
    pipeline = teacher.parents[1]
    response_path = pipeline / "blind_predictions/teacher_responses.jsonl"
    response_rows = [
        json.loads(line) for line in response_path.read_text().splitlines()
    ]
    content = json.loads(response_rows[0]["content"])
    content["reasoning"] = "Inflation evidence included twenty five basis points."
    response_rows[0]["content"] = json.dumps(content, separators=(",", ":"))
    _write_jsonl(response_path, response_rows)
    prediction_path = pipeline / "blind_predictions/predictions.jsonl"
    prediction_rows = [
        json.loads(line) for line in prediction_path.read_text().splitlines()
    ]
    prediction_rows[0]["reasoning"] = content["reasoning"]
    _write_jsonl(prediction_path, prediction_rows)
    with pytest.raises(
        augmented.Pre2009ReleaseError,
        match="v2 provider-to-prediction replay drift",
    ):
        augmented._verify_supplement_teacher(teacher)


def test_real_core_publish_verify_eval_inheritance_create_only_and_tamper(
    tmp_path: Path,
) -> None:
    teacher = _supplement_teacher(tmp_path / "supplement_teacher")
    destination = tmp_path / augmented.RELEASE_ID
    result = augmented.publish(
        core_release=augmented.DEFAULT_CORE_RELEASE,
        supplement_teacher=teacher,
        destination=destination,
        expected_core_manifest_sha256=augmented.DEFAULT_CORE_MANIFEST_SHA256,
    )
    try:
        assert result["unique_train_rows"] == 211
        assert result["physical_train_rows"] == 312
        assert result["optimizer_steps"] == 39
        verified = augmented.verify_release(
            destination,
            expected_manifest_sha256=result["release_manifest_sha256"],
        )
        with pytest.raises(
            augmented.Pre2009ReleaseError, match="invalid expected manifest SHA"
        ):
            augmented.verify_release(
                destination,
                expected_manifest_sha256=None,  # type: ignore[arg-type]
            )
        assert verified["sampler_contract"]["max_source_repeat"] == 4
        for wrapper in (
            augmented.SUPPLEMENT_BLIND_WRAPPER,
            augmented.SUPPLEMENT_TEACHER_WRAPPER,
        ):
            snapshot = (
                destination
                / "provenance/supplement_pipeline"
                / augmented.SUPPLEMENT_WRAPPER_SNAPSHOT_DIR
                / wrapper.name
            )
            assert snapshot.read_bytes() == wrapper.read_bytes()
        for role in augmented.ROLES:
            for split in augmented.EVALUATION_SPLITS:
                assert (destination / role / f"{split}.jsonl").read_bytes() == (
                    augmented.DEFAULT_CORE_RELEASE / role / f"{split}.jsonl"
                ).read_bytes()
        for family in ("unique", "repeats"):
            for split in augmented.EVALUATION_SPLITS:
                assert (
                    destination / "manifests" / family / f"{split}.jsonl"
                ).read_bytes() == (
                    augmented.DEFAULT_CORE_RELEASE
                    / "manifests"
                    / family
                    / f"{split}.jsonl"
                ).read_bytes()

        manifest_before = (destination / "release_manifest.json").read_bytes()
        with pytest.raises(
            augmented.Pre2009ReleaseError,
            match="immutable destination already exists",
        ):
            augmented.publish(
                core_release=augmented.DEFAULT_CORE_RELEASE,
                supplement_teacher=teacher,
                destination=destination,
                expected_core_manifest_sha256=augmented.DEFAULT_CORE_MANIFEST_SHA256,
            )
        assert (destination / "release_manifest.json").read_bytes() == manifest_before

        # Even a manifest+handoff-consistent metadata rewrite must fail against
        # the externally pinned manifest digest.  Runtime verification never
        # accepts an unpinned release.
        handoff_before = (destination / "handoff.json").read_bytes()
        augmented._unseal_tree(destination)
        manifest_path = destination / "release_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["created_at_utc"] = "2099-01-01T00:00:00Z"
        _write_json(manifest_path, manifest)
        rewritten_sha = augmented._sha256_file(manifest_path)
        handoff_path = destination / "handoff.json"
        handoff = json.loads(handoff_path.read_text())
        handoff["release_manifest_sha256"] = rewritten_sha
        _write_json(handoff_path, handoff)
        augmented._seal_tree(destination)
        with pytest.raises(augmented.Pre2009ReleaseError, match="manifest SHA drift"):
            augmented.verify_release(
                destination,
                expected_manifest_sha256=result["release_manifest_sha256"],
            )
        augmented._unseal_tree(destination)
        manifest_path.write_bytes(manifest_before)
        handoff_path.write_bytes(handoff_before)
        augmented._seal_tree(destination)
        augmented.verify_release(
            destination,
            expected_manifest_sha256=result["release_manifest_sha256"],
        )

        augmented._unseal_tree(destination)
        wrapper_snapshot = (
            destination
            / "provenance/supplement_pipeline"
            / augmented.SUPPLEMENT_WRAPPER_SNAPSHOT_DIR
            / augmented.SUPPLEMENT_BLIND_WRAPPER_FILE
        )
        wrapper_before = wrapper_snapshot.read_bytes()
        wrapper_snapshot.write_bytes(
            bytes([wrapper_before[0] ^ 1]) + wrapper_before[1:]
        )
        augmented._seal_tree(destination)
        with pytest.raises(augmented.Pre2009ReleaseError, match="hash drift"):
            augmented.verify_release(
                destination,
                expected_manifest_sha256=result["release_manifest_sha256"],
            )
        augmented._unseal_tree(destination)
        wrapper_snapshot.write_bytes(wrapper_before)
        augmented._seal_tree(destination)
        augmented.verify_release(
            destination,
            expected_manifest_sha256=result["release_manifest_sha256"],
        )

        augmented._unseal_tree(destination)
        schedule_path = destination / "manifests/sampler_schedule.jsonl"
        rows = [json.loads(line) for line in schedule_path.read_text().splitlines()]
        rows[0]["prompt_sha256"] = "0" * 64
        _write_jsonl(schedule_path, rows)
        augmented._seal_tree(destination)
        with pytest.raises(augmented.Pre2009ReleaseError, match="hash drift"):
            augmented.verify_release(
                destination,
                expected_manifest_sha256=result["release_manifest_sha256"],
            )
    finally:
        augmented._unseal_tree(destination)


def test_cross_population_overlap_is_rejected() -> None:
    core = [_source_row(1, "hold", 0)]
    supplement = [_source_row(2, "hike", 25)]
    supplement[0]["meeting_date"] = core[0]["meeting_date"]
    with pytest.raises(augmented.Pre2009ReleaseError, match="meeting_date overlap"):
        augmented._validate_cross_population(core, supplement)


def test_runtime_dispatch_routes_both_pre2009_roles_and_keeps_test_sealed(
    tmp_path: Path,
) -> None:
    teacher = _supplement_teacher(tmp_path / "runtime-supplement")
    destination = tmp_path / augmented.RELEASE_ID
    published = augmented.publish(
        core_release=augmented.DEFAULT_CORE_RELEASE,
        supplement_teacher=teacher,
        destination=destination,
        expected_core_manifest_sha256=augmented.DEFAULT_CORE_MANIFEST_SHA256,
    )
    core_manifest = json.loads(
        (augmented.DEFAULT_CORE_RELEASE / "release_manifest.json").read_text()
    )
    model_path = Path(core_manifest["sources"]["tokenizer"]["path"])
    try:
        for role, physical in (
            (CHK4_PRE2009_SFT_ROLE, "decision_sft"),
            (CHK4_PRE2009_GRPO_ROLE, "decision_grpo"),
        ):
            verified = verify_chk4_release_for_role(
                dataset_dir=destination / physical,
                manifest_path=destination / "release_manifest.json",
                expected_manifest_sha256=published["release_manifest_sha256"],
                dataset_role=role,
                system_prompt=augmented.STUDENT_SYSTEM_PROMPT,
                model_path=model_path,
            )
            assert verified["schema_version"] == (
                "chk4-pre2009-augmented-runtime-binding-v1"
            )
            assert verified["dataset_role"] == role
            assert verified["physical_dataset_role"] == physical
            assert set(verified["split_files"]) == {"train", "validation"}
            assert all(
                "test.jsonl" not in str(path)
                for path in verified["split_files"].values()
            )
            assert verified["test_verified_but_not_loaded"] is True
            assert verified["sampler_contract"]["type"] == (
                "manifest_fixed_schedule_v2"
            )
            assert verified["sampler_contract"]["schedule_rows"] == 312
            assert verified["sampler_contract"]["optimizer_steps"] == 39

        with pytest.raises(
            DatasetReleaseValidationError, match="does not match its logical"
        ):
            verify_chk4_release_for_role(
                dataset_dir=destination / "decision_grpo",
                manifest_path=destination / "release_manifest.json",
                expected_manifest_sha256=published["release_manifest_sha256"],
                dataset_role=CHK4_PRE2009_SFT_ROLE,
                system_prompt=augmented.STUDENT_SYSTEM_PROMPT,
                model_path=model_path,
            )
    finally:
        augmented._unseal_tree(destination)
