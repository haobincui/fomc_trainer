from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jobs.eval import prepare_chk3_core8_loo_vllm_k5 as subject
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


class _Tokenizer:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs,
    ) -> list[int]:
        assert messages[0]["role"] == "system"
        assert tokenize is True
        assert add_generation_prompt is True
        assert kwargs == {"truncation": False, "return_dict": False}
        return list(self.token_ids)


def _sample(*, prompt: str, prompt_token_count: int) -> dict:
    return {
        "sample_id": "core8-loo::meeting-0::full::none",
        "meeting_id": "meeting-0",
        "arm": "full",
        "intervention_topic": None,
        "variant_rank": 0,
        "prompt_sha256": subject._sha256_text(prompt),
        "prompt_token_count": prompt_token_count,
    }


def test_frozen_source_and_k5_budget_contract() -> None:
    path = subject.DEFAULT_SOURCE_SAMPLE_MANIFEST
    assert subject._sha256_file(path) == subject.SOURCE_SAMPLE_MANIFEST_SHA256
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert len(manifest["samples"]) == subject.EXPECTED_PROMPTS == 2_176
    assert manifest["selection"]["meetings"] == subject.EXPECTED_MEETINGS == 128
    assert manifest["selection"]["variants_per_meeting"] == 17
    assert subject.REPLICATE_SEEDS == (
        20260811,
        21260811,
        22260811,
        23260811,
        24260811,
    )
    assert subject.EXPECTED_CASES == 10_880
    assert subject.MAX_PROMPT_TOKENS == 1_536
    maximum = max(int(row["prompt_token_count"]) for row in manifest["samples"])
    assert maximum == 1_339
    assert maximum + subject.MAX_NEW_TOKENS == 3_899 <= subject.MAX_MODEL_LEN
    assert subject.DEFAULT_OUTPUT_DIR.name == "preparation_v2"
    assert subject.COHORT_SCHEMA.endswith("-v2")
    assert subject.PREPARATION_SCHEMA.endswith("-v2")


def test_exact_token_ledger_persists_ids_and_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "EXPECTED_PROMPTS", 1)
    prompt = "frozen prompt"
    sample = _sample(prompt=prompt, prompt_token_count=3)
    rows = subject.build_token_ledger(
        sample_manifest={"samples": [sample]},
        bound_rows={sample["sample_id"]: {"prompt": prompt}},
        tokenizer=_Tokenizer([128000, 42, 128014]),
        system_prompt="system",
        user_prompt_suffix=None,
    )
    assert rows == [
        {
            "schema_version": subject.LEDGER_ROW_SCHEMA,
            "absolute_prompt_index": 0,
            "source_sample_line_number": 1,
            "sample_id": sample["sample_id"],
            "meeting_id": "meeting-0",
            "arm": "full",
            "intervention_topic": None,
            "variant_rank": 0,
            "prompt_sha256": subject._sha256_text(prompt),
            "messages_sha256": rows[0]["messages_sha256"],
            "prompt_token_count": 3,
            "prompt_token_ids": [128000, 42, 128014],
            "prompt_token_ids_sha256": subject._sha256_text(
                subject._canonical([128000, 42, 128014])
            ),
        }
    ]


def test_token_ledger_rejects_prompt_budget_over_4096(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "EXPECTED_PROMPTS", 1)
    prompt = "over budget"
    ids = list(range(subject.MAX_PROMPT_TOKENS + 1))
    sample = _sample(prompt=prompt, prompt_token_count=len(ids))
    with pytest.raises(subject.LooVllmK5PreparationError, match="exceeds 4096"):
        subject.build_token_ledger(
            sample_manifest={"samples": [sample]},
            bound_rows={sample["sample_id"]: {"prompt": prompt}},
            tokenizer=_Tokenizer(ids),
            system_prompt="system",
            user_prompt_suffix=None,
        )


def test_execution_policy_freezes_user_approved_schedule_waiver() -> None:
    excluded = {
        "role": "excluded_historical_greedy_k1_evidence_only",
        "generation_manifest": {"sha256": "b" * 64},
        "generations": {"sha256": "c" * 64},
        "score_manifest": {"sha256": "d" * 64},
    }
    policy = subject.build_execution_policy(
        source_sample_manifest={"path": "/source", "sha256": "a" * 64},
        model_runtime_inventory={
            "path": "/model",
            "files": {"chat_template.jinja": subject.EXPECTED_CHAT_TEMPLATE_SHA256},
        },
        excluded_historical_greedy_k1=excluded,
    )
    validate_manifest_integrity(policy)
    assert policy["sealed"] is policy["immutable"] is True
    assert policy["topology"] == {
        "workers": 2,
        "worker_type": "two_independent_full_model_vllm_instances",
        "physical_gpu_indexes": [0, 1],
        "data_parallel_size_per_worker": 1,
        "tensor_parallel_size_per_worker": 1,
        "pipeline_parallel_size_per_worker": 1,
        "cross_worker_data_parallel_collective": False,
        "dtype": "bfloat16",
        "quantization": None,
    }
    assert policy["engine"]["max_num_seqs_per_worker"] == 16
    assert policy["engine"]["gpu_memory_utilization_per_worker"] == 0.95
    assert policy["sampling"]["replicates"] == 5
    assert policy["sampling"]["all_replicates_fresh_vllm_generations"] is True
    assert policy["sampling"]["historical_greedy_k1_excluded"] is True
    replay = policy["scheduling_and_replay"]
    assert replay["schedule_sensitive_token_identity_replay_gate"] == "non_blocking"
    assert replay["token_identity_across_launches_or_batch_schedules_required"] is False
    assert replay["canonical_case_order"] == (
        "meeting_order_then_variant_rank_then_replicate_id"
    )
    assert replay["absolute_prompt_index"] == "absolute_case_index//5"
    assert replay["replicate_id"] == "absolute_case_index%5"
    assert replay["shard_unit"] == "meeting_replicate_17_arm_block"
    assert replay["shard_function"] == "paired_block_mod_2"
    assert replay["shard_id"] == "(meeting_index*5+replicate_id)%2"
    assert policy["excluded_historical_greedy_k1"] == excluded


def test_sample_major_index_roundtrip_keeps_paired_arms_on_one_shard() -> None:
    paired_shards: dict[tuple[int, int], set[int]] = {}
    paired_counts: dict[tuple[int, int], int] = {}
    for absolute_case_index in range(subject.EXPECTED_CASES):
        case = subject.canonical_case_coordinates(absolute_case_index)
        assert case["absolute_prompt_index"] == absolute_case_index // 5
        assert case["replicate_id"] == absolute_case_index % 5
        assert case["meeting_index"] == case["absolute_prompt_index"] // 17
        assert case["variant_rank"] == case["absolute_prompt_index"] % 17
        assert case["shard_id"] == (
            case["meeting_index"] * 5 + case["replicate_id"]
        ) % 2
        key = (case["meeting_index"], case["replicate_id"])
        paired_shards.setdefault(key, set()).add(case["shard_id"])
        paired_counts[key] = paired_counts.get(key, 0) + 1
    assert len(paired_shards) == 128 * 5
    assert all(len(shards) == 1 for shards in paired_shards.values())
    assert set(paired_counts.values()) == {17}


def test_historical_greedy_k1_is_bound_as_excluded_evidence() -> None:
    evidence = subject.load_excluded_historical_greedy_evidence()
    assert evidence["role"] == "excluded_historical_greedy_k1_evidence_only"
    assert evidence["included_in_stochastic_replicates"] is False
    assert evidence["generation_manifest"]["sha256"] == (
        subject.HISTORICAL_GREEDY_GENERATION_MANIFEST_SHA256
    )
    assert evidence["generations"]["sha256"] == (
        subject.HISTORICAL_GREEDY_GENERATIONS_SHA256
    )
    assert evidence["generations"]["rows"] == 2_176
    assert evidence["score_manifest"]["sha256"] == (
        subject.HISTORICAL_GREEDY_SCORE_MANIFEST_SHA256
    )


def test_prepare_writes_new_sealed_cohort_policy_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = "fixture prompt"
    source_path = tmp_path / "source.json"
    source_path.write_text('{"fixture":true}\n', encoding="utf-8")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    sample = _sample(prompt=prompt, prompt_token_count=3)
    manifest = {
        "integrity": {"payload_sha256": "b" * 64},
        "task_contract_id": "chk3-native-analysis-to-minutes-v1",
        "selection": {"rows": 1, "meetings": 1, "variants_per_meeting": 1},
        "samples": [sample],
        "dataset": {"path": "/dataset", "sha256": "c" * 64, "rows": 1},
        "source_release": {"path": "/release", "sha256": "d" * 64},
        "source_panel": {"path": "/panel", "sha256": "e" * 64},
        "model_anchor": {"path": "/anchor", "sha256": "f" * 64},
        "exact_merge_evidence": {"path": "/merge", "sha256": "1" * 64},
        "prompt_contract": {"training_config": "/config"},
        "tokenizer": {"path": "/model", "files": {"tokenizer.json": "2" * 64}},
    }
    input_rows = [{**sample, "prompt": prompt}]
    runtime_inventory = {
        "path": "/model",
        "files": {"chat_template.jinja": subject.EXPECTED_CHAT_TEMPLATE_SHA256},
        "file_count": 1,
        "file_hash_inventory_sha256": "3" * 64,
        "chat_template_jinja_sha256": subject.EXPECTED_CHAT_TEMPLATE_SHA256,
    }
    monkeypatch.setattr(subject, "SOURCE_SAMPLE_MANIFEST_SHA256", source_sha)
    monkeypatch.setattr(subject, "EXPECTED_PROMPTS", 1)
    monkeypatch.setattr(subject, "EXPECTED_MEETINGS", 1)
    monkeypatch.setattr(subject, "VARIANTS_PER_MEETING", 1)
    monkeypatch.setattr(subject, "EXPECTED_CASES", 5)
    monkeypatch.setattr(subject, "ABSOLUTE_CHUNK_SIZE", 5)
    monkeypatch.setattr(subject, "EXPECTED_CHUNKS", 1)
    monkeypatch.setattr(
        subject,
        "load_source_inputs",
        lambda path, expected: (
            manifest,
            input_rows,
            _Tokenizer([128000, 42, 128014]),
            source_sha,
        ),
    )
    monkeypatch.setattr(
        subject,
        "_load_prompt_contract",
        lambda loaded_manifest, tokenizer: ("system", None, {"eos_token_id": 1}),
    )
    monkeypatch.setattr(
        subject,
        "load_model_runtime_inventory",
        lambda loaded_manifest: runtime_inventory,
    )
    excluded = {
        "role": "excluded_historical_greedy_k1_evidence_only",
        "included_in_stochastic_replicates": False,
        "generation_manifest": {"sha256": "4" * 64},
        "generations": {"sha256": "5" * 64, "rows": 1},
        "score_manifest": {"sha256": "6" * 64},
    }
    monkeypatch.setattr(
        subject,
        "load_excluded_historical_greedy_evidence",
        lambda: excluded,
    )

    output_dir = tmp_path / "prepared"
    result = subject.prepare(
        source_sample_manifest_path=source_path,
        source_sample_manifest_sha256=source_sha,
        output_dir=output_dir,
    )
    validate_manifest_integrity(result)
    cohort = json.loads((output_dir / subject.COHORT_FILENAME).read_text())
    policy = json.loads(
        (output_dir / subject.EXECUTION_POLICY_FILENAME).read_text()
    )
    validate_manifest_integrity(cohort)
    validate_manifest_integrity(policy)
    assert cohort["token_ledger"]["rows"] == 1
    assert cohort["token_ledger"]["full_prompt_token_ids_persisted"] is True
    assert cohort["token_ledger"]["vllm_runtime_chat_templating"] is False
    assert cohort["generation_design"]["canonical_case_order"] == (
        "meeting_order_then_variant_rank_then_replicate_id"
    )
    assert cohort["excluded_historical_greedy_k1"] == excluded
    assert cohort["execution_policy"]["sha256"] == subject._sha256_file(
        output_dir / subject.EXECUTION_POLICY_FILENAME
    )
    assert result["cohort"]["sha256"] == subject._sha256_file(
        output_dir / subject.COHORT_FILENAME
    )
    ledger_line = (output_dir / subject.LEDGER_FILENAME).read_text().splitlines()
    assert len(ledger_line) == 1
    assert json.loads(ledger_line[0])["prompt_token_ids"] == [128000, 42, 128014]
    with pytest.raises(subject.LooVllmK5PreparationError, match="already exists"):
        subject.prepare(
            source_sample_manifest_path=source_path,
            source_sample_manifest_sha256=source_sha,
            output_dir=output_dir,
        )
