from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2 import publish_chk4_training_data as publisher


def _unique_responses() -> dict[str, tuple[str, str]]:
    output: dict[str, tuple[str, str]] = {}
    for split in publisher.SPLITS:
        unique_rows = publisher._read_jsonl(
            publisher.DEFAULT_SOURCE / "manifests/unique" / f"{split}.jsonl",
            label=f"{split} unique test fixture",
        )
        sft_rows = publisher._read_jsonl(
            publisher.DEFAULT_SOURCE / "decision_sft" / f"{split}.jsonl",
            label=f"{split} SFT test fixture",
        )
        offset = 0
        for unique in unique_rows:
            output[unique["sample_id"]] = (split, sft_rows[offset]["response"])
            offset += unique["repeat_factor"]
        assert offset == len(sft_rows)
    return output


def test_all_reasoning_repairs_remove_explicit_numbers_and_dates() -> None:
    source = _unique_responses()
    assert set(source) >= set(publisher.REASONING_REPAIRS)
    assert len(publisher.REASONING_REPAIRS) == 23
    for sample_id in publisher.REASONING_REPAIRS:
        split, response = source[sample_id]
        updated, record = publisher._apply_repairs(
            source_id=sample_id,
            response=response,
            split=split,
        )
        assert record is not None
        assert record["old_response_sha256"] != record["new_response_sha256"]
        assert record["prompt_and_final_decision_unchanged"] is True
        publisher._parse_response(updated, label=sample_id)


@pytest.mark.parametrize(
    "analysis",
    [
        "The Committee raised the target range by 25 basis points.",
        "The target range was increased by 25 basis points.",
        "Policymakers opted to lower rates by 50 basis points.",
        "The FOMC cut rates by 25 bp.",
        "Officials left rates unchanged.",
        "The vote resulted in a 50 basis point cut.",
        "No change was made to the target range.",
        'The target was {"direction":"cut","magnitude_bp":25}.',
        "Outcome: hike by 25 bp.",
        "At the target meeting, rates were raised by 25 basis points.",
        "A 25 basis point rate reduction was announced.",
        "direction: cut; magnitude_bp: 25",
        "Committee chose to keep rates unchanged.",
        "A hold was warranted.",
        "Rates would remain unchanged.",
        "Inflation rose in December, and the Committee raised rates by 25 basis points.",
        "Following weak data, the Committee cut rates by 50 bp.",
        "The Committee previously reviewed inflation and raised rates by 25 bp.",
        "At the prior meeting the Committee held rates, but today it cut rates by 50 bp.",
        "The Fed raised rates by 25 basis points.",
        "Central bank raised policy rates by 25 bp.",
        "Rates rose by 25 bp.",
        "A quarter-point increase was approved.",
        "Hike 25 bp.",
        "direction = cut; magnitude_bp = 25",
        "At the prior meeting the Committee held rates and the FOMC cut rates by 50 bp.",
        "In December 2023 data weakened and in January the Committee raised rates by 25 bp.",
    ],
)
def test_target_outcome_language_is_rejected(analysis: str) -> None:
    prompt = publisher.USER_PROMPT_PREFIX + json.dumps({"analysis": analysis})
    with pytest.raises(publisher.Chk4ReleaseError, match="target-meeting outcome"):
        publisher._parse_prompt(
            prompt,
            meeting_date="2024-01-31",
            label="rejected",
        )


@pytest.mark.parametrize(
    "analysis",
    [
        (
            "Before the meeting, the effective federal funds rate remained within "
            "the previously announced target range. Inflation stayed elevated."
        ),
        "At the prior meeting, the Committee voted to raise rates.",
        "Higher tariffs would raise inflation.",
        "Weaker demand would cut inflation pressure.",
        "A 25 basis point rate cut was announced at the prior meeting.",
        "The Federal Reserve increased the federal funds rate target range in December.",
        "The Federal Reserve raised the target range for the federal funds rate in March.",
        "The Federal Reserve reduced the federal funds rate to near zero by late March.",
        "Officials held differing views about inflation risks.",
        "Policymakers decided that inflation remained elevated.",
        "Committee left its economic assessment unchanged.",
        "Committee raised its inflation projection.",
        "FOMC increased its estimate of potential output.",
    ],
)
def test_pre_meeting_policy_state_is_allowed(analysis: str) -> None:
    prefix = publisher.USER_PROMPT_PREFIX
    allowed = prefix + json.dumps({"analysis": analysis})
    assert publisher._parse_prompt(
        allowed,
        meeting_date="2024-01-31",
        label="allowed",
        point_in_time_attested=True,
    ) == analysis


@pytest.mark.parametrize(
    "rendered_date",
    ["2024-01-31", "January 31, 2024", "Jan 31, 2024", "01/31/2024", "1-31-2024"],
)
def test_exact_target_date_variants_are_rejected(rendered_date: str) -> None:
    prompt = publisher.USER_PROMPT_PREFIX + json.dumps(
        {"analysis": f"The evidence cutoff is {rendered_date}."}
    )
    with pytest.raises(publisher.Chk4ReleaseError, match="meeting_date"):
        publisher._parse_prompt(
            prompt,
            meeting_date="2024-01-31",
            label="date leak",
            point_in_time_attested=True,
        )


@pytest.mark.parametrize(
    "analysis",
    [
        "In January, the Committee raised rates by 25 basis points.",
        "In 2024, the FOMC cut rates by 50 bp.",
    ],
)
def test_same_period_policy_action_is_rejected_even_when_lineage_attested(
    analysis: str,
) -> None:
    prompt = publisher.USER_PROMPT_PREFIX + json.dumps({"analysis": analysis})
    with pytest.raises(publisher.Chk4ReleaseError, match="target-meeting outcome"):
        publisher._parse_prompt(
            prompt,
            meeting_date="2024-01-31",
            label="ambiguous same-period action",
            point_in_time_attested=True,
        )


def test_contract_has_a_valid_self_digest() -> None:
    source_info = publisher._validate_source_summaries(
        source=publisher.DEFAULT_SOURCE,
        teacher=publisher.DEFAULT_TEACHER,
        briefs=publisher.DEFAULT_BRIEFS,
    )
    contract = publisher._build_input_contract(source_info)
    digest = contract.pop("contract_sha256")
    assert digest == publisher._sha256_text(publisher._canonical_json(contract))
    assert contract["population_scope"] == "core-only; supplement_admitted=0"
    assert "inference-proof" in contract["identity_inference_risk"]
    assert "direct_sample_id_or_exact_target_meeting_date_field" in contract[
        "forbidden_target_fields"
    ]
    assert "sample_or_meeting_identity" not in contract["forbidden_target_fields"]


def test_real_release_publish_and_replay_verification(tmp_path: Path) -> None:
    destination = tmp_path / "chk4_decision_warmstart_grpo_core_test"
    result = publisher.publish(
        source=publisher.DEFAULT_SOURCE,
        teacher=publisher.DEFAULT_TEACHER,
        briefs=publisher.DEFAULT_BRIEFS,
        tokenizer=publisher.DEFAULT_TOKENIZER,
        destination=destination,
    )
    assert result["status"] == "published"
    manifest = publisher.verify_release(destination)
    assert manifest["quality_status"] == "passed"
    assert manifest["population_scope"] == "core-only"
    assert manifest["target_blindness"] == "target-decision-blind"
    audit = json.loads(
        (destination / "audits/data_quality.json").read_text(encoding="utf-8")
    )
    assert audit["reasoning_repairs"] == 23
    assert audit["reasoning_digit_or_date_violations"] == 0
    assert audit["evaluation_meetings_with_action_absent_from_train"] == 7
    assert all(
        stats["sft_overflow_rows"] == 0
        and stats["grpo_prompt_overflow_rows"] == 0
        for stats in audit["split_token_stats"].values()
    )
    point_in_time = json.loads(
        (destination / "audits/point_in_time_lineage.json").read_text(
            encoding="utf-8"
        )
    )
    assert point_in_time["canonical_chk1"]["source_rows"] == 2072
    assert point_in_time["teacher_target_rows_bound"] == 128
    assert point_in_time["teacher_brief_rows_bound"] == 128
    assert point_in_time["canonical_chk1"]["future_evidence_violations"] == 0
    assert point_in_time["canonical_chk1"]["selected_evidence_id_binding"] == {
        "empty": 50,
        "exact": 2003,
        "unique_prefix": 11,
        "unresolved": 8,
    }
    manifest_sha = publisher._sha256_file(destination / "release_manifest.json")
    publisher.verify_release(
        destination, expected_manifest_sha256=manifest_sha
    )

    publisher._unseal_staging(destination)
    extra_path = destination / "extra.json"
    extra_path.write_text("{}\n", encoding="utf-8")
    publisher._seal_tree(destination)
    with pytest.raises(publisher.Chk4ReleaseError, match="unlisted files"):
        publisher.verify_release(
            destination, expected_manifest_sha256=manifest_sha
        )
    publisher._unseal_staging(destination)
    extra_path.unlink()
    publisher._seal_tree(destination)

    # The handoff is outside the manifest file table because it contains the
    # manifest SHA, so its unsigned payload must still be manifest-bound.
    publisher._unseal_staging(destination)
    handoff_path = destination / "handoff.json"
    original_handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    tampered_handoff = dict(original_handoff)
    tampered_handoff["decision_sft_path"] = "unverified_sft"
    handoff_path.write_text(
        json.dumps(tampered_handoff, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    publisher._seal_tree(destination)
    with pytest.raises(publisher.Chk4ReleaseError, match="handoff payload binding"):
        publisher.verify_release(
            destination, expected_manifest_sha256=manifest_sha
        )

    # A self-consistent replacement of an SFT rationale must still fail its
    # raw-teacher -> teacher-SFT -> deterministic-repair lineage replay.
    publisher._unseal_staging(destination)
    manifest_path = destination / "release_manifest.json"
    handoff_path.write_text(
        json.dumps(original_handoff, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    original_manifest_bytes = manifest_path.read_bytes()
    original_handoff_bytes = handoff_path.read_bytes()
    sft_path = destination / "decision_sft/train.jsonl"
    unique_path = destination / "manifests/unique/train.jsonl"
    original_sft_bytes = sft_path.read_bytes()
    original_unique_bytes = unique_path.read_bytes()
    sft_rows = publisher._read_jsonl(sft_path, label="SFT rationale tamper fixture")
    unique_rows = publisher._read_jsonl(unique_path, label="unique tamper fixture")
    _reasoning, final_decision = sft_rows[0]["response"].split(publisher.BOUNDARY, 1)
    replacement = (
        "The available evidence points to a deliberately substituted qualitative rationale."
        + publisher.BOUNDARY
        + final_decision
    )
    for index in range(unique_rows[0]["repeat_factor"]):
        sft_rows[index]["response"] = replacement
    unique_rows[0]["response_sha256"] = publisher._sha256_text(replacement)
    sft_path.write_text(
        "".join(publisher._canonical_json(row) + "\n" for row in sft_rows),
        encoding="utf-8",
    )
    unique_path.write_text(
        "".join(publisher._canonical_json(row) + "\n" for row in unique_rows),
        encoding="utf-8",
    )
    rationale_manifest = json.loads(original_manifest_bytes)
    for relative, path in (
        ("decision_sft/train.jsonl", sft_path),
        ("manifests/unique/train.jsonl", unique_path),
    ):
        rationale_manifest["files"][relative]["bytes"] = path.stat().st_size
        rationale_manifest["files"][relative]["sha256"] = publisher._sha256_file(path)
    manifest_path.write_text(
        json.dumps(rationale_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rationale_handoff = dict(original_handoff)
    rationale_handoff["release_manifest_sha256"] = publisher._sha256_file(manifest_path)
    handoff_path.write_text(
        json.dumps(rationale_handoff, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    publisher._seal_tree(destination)
    with pytest.raises(
        publisher.Chk4ReleaseError,
        match="clean SFT/teacher target response drift",
    ):
        publisher.verify_release(destination)

    publisher._unseal_staging(destination)
    manifest_path.write_bytes(original_manifest_bytes)
    handoff_path.write_bytes(original_handoff_bytes)
    sft_path.write_bytes(original_sft_bytes)
    unique_path.write_bytes(original_unique_bytes)
    publisher._seal_tree(destination)
    publisher.verify_release(destination, expected_manifest_sha256=manifest_sha)

    # Reproduce the old self-consistent-tamper attack: modify a GRPO label and
    # update all internal file hashes. Semantic replay must still reject it.
    publisher._unseal_staging(destination)
    handoff_path.write_text(
        json.dumps(original_handoff, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    grpo_path = destination / "decision_grpo/train.jsonl"
    grpo_rows = publisher._read_jsonl(grpo_path, label="tamper fixture")
    grpo_rows[0]["direction"] = "hold"
    grpo_rows[0]["magnitude_bp"] = 0
    grpo_path.write_text(
        "".join(publisher._canonical_json(row) + "\n" for row in grpo_rows),
        encoding="utf-8",
    )
    tampered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = tampered_manifest["files"]["decision_grpo/train.jsonl"]
    record["bytes"] = grpo_path.stat().st_size
    record["sha256"] = publisher._sha256_file(grpo_path)
    manifest_path.write_text(
        json.dumps(tampered_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    original_handoff["release_manifest_sha256"] = publisher._sha256_file(
        manifest_path
    )
    handoff_path.write_text(
        json.dumps(original_handoff, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    publisher._seal_tree(destination)
    with pytest.raises(publisher.Chk4ReleaseError, match="externally pinned SHA"):
        publisher.verify_release(
            destination, expected_manifest_sha256=manifest_sha
        )
    with pytest.raises(publisher.Chk4ReleaseError):
        publisher.verify_release(destination)
    publisher._unseal_staging(destination)
