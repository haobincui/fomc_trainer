from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jobs.eval import prepare_chk3_core8_loo_vllm_k10_increment as subject
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


class _Tokenizer:
    name_or_path = "/model"
    eos_token_id = 1
    pad_token_id = 1
    bos_token_id = None
    chat_template = "template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "truncation": False,
            "return_dict": False,
        }
        return [128000, 42, 128014]


def _sample() -> dict:
    prompt = "fixture prompt"
    return {
        "sample_id": "core8-loo::meeting-0::full::none",
        "meeting_id": "meeting-0",
        "arm": "full",
        "intervention_topic": None,
        "variant_rank": 0,
        "prompt_sha256": subject.frozen._sha256_text(prompt),
        "prompt_token_count": 3,
    }


def test_increment_identity_seed_and_output_contract() -> None:
    assert subject.REPLICATE_SEEDS == (
        25260811,
        26260811,
        27260811,
        28260811,
        29260811,
    )
    assert subject.LOCAL_REPLICATE_IDS == (0, 1, 2, 3, 4)
    assert subject.GLOBAL_REPLICATE_OFFSET == 5
    assert subject.GLOBAL_REPLICATE_IDS == (5, 6, 7, 8, 9)
    assert subject.EXPECTED_CASES == 10_880
    assert subject.DEFAULT_OUTPUT_DIR.as_posix().endswith(
        "chk3_cp318_core8_loo_vllm_k10_n128_1993_2008_20260824_v1/"
        "preparation_increment_k6_k10_v1"
    )
    contract = subject.replicate_indexing_contract()
    assert contract["replicate_seed_by_global_id"] == {
        "5": 25260811,
        "6": 26260811,
        "7": 27260811,
        "8": 28260811,
        "9": 29260811,
    }
    assert contract["prior_k5_rows_reused_in_increment"] is False


def test_local_coordinates_expose_sealed_global_ids() -> None:
    first = subject.canonical_case_coordinates(0)
    fifth = subject.canonical_case_coordinates(4)
    sixth = subject.canonical_case_coordinates(5)
    assert (first["replicate_id"], first["global_replicate_id"]) == (0, 5)
    assert (fifth["replicate_id"], fifth["global_replicate_id"]) == (4, 9)
    assert (sixth["replicate_id"], sixth["global_replicate_id"]) == (0, 5)
    assert sixth["absolute_prompt_index"] == 1


def test_predecessor_and_abandoned_nf4_artifacts_are_bound_and_excluded() -> None:
    prior = subject.prior_k5_bindings()
    assert prior["cohort"]["sha256"] == subject.PRIOR_K5_COHORT_SHA256
    assert prior["execution_policy"]["sha256"] == subject.PRIOR_K5_POLICY_SHA256
    abandoned = subject.excluded_incomplete_nf4_k10_bindings()
    assert abandoned["included_in_increment"] is False
    assert abandoned["included_in_combined_k10"] is False
    assert abandoned["artifacts"]["partial_generations"]["rows"] == 2_071
    assert subject.increment_authorization()["authorization_date"] == "2026-08-24"


def test_prepare_is_sealed_create_only_and_binds_adapter_and_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = _sample()
    prompt = "fixture prompt"
    source_path = tmp_path / "source.json"
    source_path.write_text('{"fixture":true}\n', encoding="utf-8")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
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
        "files": {"chat_template.jinja": subject.frozen.EXPECTED_CHAT_TEMPLATE_SHA256},
        "file_count": 1,
        "file_hash_inventory_sha256": "3" * 64,
        "chat_template_jinja_sha256": subject.frozen.EXPECTED_CHAT_TEMPLATE_SHA256,
    }
    excluded_greedy = {
        "role": "excluded_historical_greedy_k1_evidence_only",
        "included_in_stochastic_replicates": False,
        "generation_manifest": {"sha256": "4" * 64},
        "generations": {"sha256": "5" * 64, "rows": 1},
        "score_manifest": {"sha256": "6" * 64},
    }
    monkeypatch.setattr(subject.frozen, "SOURCE_SAMPLE_MANIFEST_SHA256", source_sha)
    monkeypatch.setattr(subject.frozen, "EXPECTED_PROMPTS", 1)
    monkeypatch.setattr(subject.frozen, "EXPECTED_MEETINGS", 1)
    monkeypatch.setattr(subject.frozen, "VARIANTS_PER_MEETING", 1)
    monkeypatch.setattr(subject, "EXPECTED_CASES", 5)
    monkeypatch.setattr(subject, "ABSOLUTE_CHUNK_SIZE", 5)
    monkeypatch.setattr(subject, "EXPECTED_CHUNKS", 1)
    monkeypatch.setattr(
        subject.frozen,
        "load_source_inputs",
        lambda path, expected: (
            manifest,
            input_rows,
            _Tokenizer(),
            source_sha,
        ),
    )
    monkeypatch.setattr(
        subject.frozen,
        "_load_prompt_contract",
        lambda loaded, tokenizer: ("system", None, {"eos_token_id": 1}),
    )
    monkeypatch.setattr(
        subject.frozen,
        "load_model_runtime_inventory",
        lambda loaded: runtime_inventory,
    )
    monkeypatch.setattr(
        subject.frozen,
        "load_excluded_historical_greedy_evidence",
        lambda: excluded_greedy,
    )

    output_dir = tmp_path / "increment"
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
    for value in (result, cohort, policy):
        validate_manifest_integrity(value)
        assert value["replicate_indexing"] == subject.replicate_indexing_contract()
        assert value["increment_authorization"] == subject.increment_authorization()
        assert value["excluded_incomplete_nf4_k10"]["included_in_increment"] is False
        assert "increment_preparer_adapter" in value["implementation"]
        assert "frozen_k5_preparer_base" in value["implementation"]
    assert cohort["generation_design"]["replicate_seeds"] == list(
        subject.REPLICATE_SEEDS
    )
    with pytest.raises(
        subject.LooVllmK10IncrementPreparationError,
        match="already exists",
    ):
        subject.prepare(
            source_sample_manifest_path=source_path,
            source_sample_manifest_sha256=source_sha,
            output_dir=output_dir,
        )
