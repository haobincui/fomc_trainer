from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jobs.eval.eval_retrain_v2_chk0_chk1_generation import (
    ARTIFACT_IDS,
    BASE_ARTIFACT_ID,
    CHK1_ARTIFACT_ID,
    EXPECTED_SAMPLE_COUNT,
    build_parser,
    validate_inputs,
)
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import derive_row_seed, seal_manifest


MEETINGS = [f"2025-{month:02d}-01" for month in range(1, 12)]
PROSPECTIVE = MEETINGS[2:]
SECTIONS = ("participants_views", "economic_situation", "financial_situation")


@pytest.fixture(autouse=True)
def _stub_deep_progress_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generator tests cover the byte-level chain; wrapper tests cover routing."""

    monkeypatch.setattr(
        "jobs.eval.eval_retrain_v2_chk0_chk1_generation."
        "validate_generation_progress_binding",
        lambda **_kwargs: {
            "contract_sha256": "f" * 64,
            "state_sha256": "1" * 64,
            "partial_sha256": "2" * 64,
            "row_count": EXPECTED_SAMPLE_COUNT,
        },
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_sealed(path: Path, payload: dict) -> None:
    _write_json(path, seal_manifest(payload))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _fixture(root: Path) -> dict[str, object]:
    model_base = root / "base-model"
    model_chk1 = root / "chk1-model"
    adapter = root / "adapter"
    bert = root / "bert"
    mpnet = root / "mpnet"
    for directory in (model_base, model_chk1, adapter, bert, mpnet):
        directory.mkdir()
        (directory / "sentinel").write_text(directory.name, encoding="utf-8")

    base_sha = "a" * 64
    chk1_sha = "b" * 64
    adapter_sha = "c" * 64
    bert_sha = "d" * 64
    mpnet_sha = "e" * 64

    config_path = root / "config.json"
    prompts_path = root / "prompts.jsonl"
    references_path = root / "references.jsonl"
    test_manifest_path = root / "test_manifest.json"
    checkpoint_manifest_path = root / "checkpoint_manifest.json"
    lineage_manifest_path = root / "lineage_manifest.json"
    semantic_manifest_path = root / "semantic_manifest.json"
    exact_merge_path = root / "exact_merge.json"
    config = {
        "schema_version": "checkpoint-generation-eval-config-v1",
        "evaluation_id": "fixture-chk0-chk1-eval",
        "population": {
            "meeting_dates": MEETINGS,
            "prospective_only_meeting_dates": PROSPECTIVE,
        },
        "sections": [{"section_id": section} for section in SECTIONS],
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "base_seed": 20260729,
            "seed_policy": "sample-id-sha256-v1",
            "max_new_tokens": 8192,
            "max_model_len": 24576,
        },
    }
    _write_json(config_path, config)

    prompts: list[dict] = []
    references: list[dict] = []
    for meeting in MEETINGS:
        for section in SECTIONS:
            sample_id = f"{meeting}::{section}"
            prompts.append(
                {
                    "sample_id": sample_id,
                    "meeting_id": meeting,
                    "section_name": section,
                    "prompt": f"Evidence for {sample_id}",
                    "prompt_sha256": hashlib.sha256(
                        f"Evidence for {sample_id}".encode("utf-8")
                    ).hexdigest(),
                }
            )
            references.append(
                {
                    "sample_id": sample_id,
                    "meeting_id": meeting,
                    "section_name": section,
                    "reference": "Inflation declined.",
                }
            )
    assert len(prompts) == EXPECTED_SAMPLE_COUNT
    _write_jsonl(prompts_path, prompts)
    _write_jsonl(references_path, references)

    _write_sealed(
        test_manifest_path,
        {
            "schema_version": "checkpoint-eval-test-manifest-v1",
            "evaluation_id": config["evaluation_id"],
            "sample_count": 33,
            "section_count": 3,
            "meeting_dates": MEETINGS,
            "prospective_only_meeting_dates": PROSPECTIVE,
            "reference_in_prompt": False,
            "secondary_llm_summary": False,
            "inputs": {
                "config": {"path": str(config_path), "sha256": sha256_file(config_path)}
            },
            "outputs": {
                "prompts": {
                    "path": str(prompts_path),
                    "sha256": sha256_file(prompts_path),
                    "row_count": 33,
                },
                "references": {
                    "path": str(references_path),
                    "sha256": sha256_file(references_path),
                    "row_count": 33,
                },
            },
        },
    )

    _write_sealed(
        checkpoint_manifest_path,
        {
            "schema_version": "retrain-v2-checkpoint-eval-manifest-v1",
            "artifacts": [
                {
                    "artifact_id": BASE_ARTIFACT_ID,
                    "design_checkpoint_id": "chk-0",
                    "intended_parent_id": None,
                    "model_path": str(model_base),
                    "model_sha256": base_sha,
                    "tokenizer_path": str(model_base),
                    "tokenizer_sha256": base_sha,
                    "usable_for_evaluation": True,
                    "verified_parent_artifact_id": None,
                },
                {
                    "artifact_id": CHK1_ARTIFACT_ID,
                    "design_checkpoint_id": "chk-1",
                    "intended_parent_id": "chk-0",
                    "model_path": str(model_chk1),
                    "model_sha256": chk1_sha,
                    "tokenizer_path": str(model_chk1),
                    "tokenizer_sha256": chk1_sha,
                    "adapter_path": str(adapter),
                    "adapter_sha256": adapter_sha,
                    "usable_for_evaluation": True,
                    "verified_parent_artifact_id": BASE_ARTIFACT_ID,
                },
            ],
        },
    )
    _write_sealed(
        exact_merge_path,
        {
            "schema_version": "lora-merge-lineage-evidence-v1",
            "algorithm_version": "peft-lora-fp32-exact-v1",
            "subject_artifact_id": "chk1-clean-v2-lr1e6-cp200",
            "conclusion": "exact_base_plus_adapter_merge_verified",
            "sources": {
                "base_model": {"path": str(model_base), "sha256": base_sha},
                "merged_model": {"path": str(model_chk1), "sha256": chk1_sha},
                "adapter": {"path": str(adapter), "sha256": adapter_sha},
            },
            "tensor_verification": {
                "model_tensor_count": 291,
                "adapted_model_tensor_count": 224,
                "exact_adapted_model_tensor_count": 224,
                "unchanged_model_tensor_count": 67,
                "exact_unchanged_model_tensor_count": 67,
                "adapter_tensor_count": 448,
                "mismatch_count": 0,
            },
        },
    )
    checkpoint = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    checkpoint.pop("integrity")
    exact_merge = json.loads(exact_merge_path.read_text(encoding="utf-8"))
    test_manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))
    checkpoint["bindings"] = {
        "evaluation_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "test_manifest": {
            "path": str(test_manifest_path),
            "sha256": sha256_file(test_manifest_path),
            "payload_sha256": test_manifest["integrity"]["payload_sha256"],
        },
        "exact_merge_evidence": {
            "path": str(exact_merge_path),
            "sha256": sha256_file(exact_merge_path),
            "payload_sha256": exact_merge["integrity"]["payload_sha256"],
        },
    }
    _write_sealed(checkpoint_manifest_path, checkpoint)
    _write_sealed(
        lineage_manifest_path,
        {
            "schema_version": "retrain-v2-eval-lineage-v1",
            "checkpoints": {
                BASE_ARTIFACT_ID: {
                    "artifact_id": BASE_ARTIFACT_ID,
                    "parent_artifact_id": None,
                },
                CHK1_ARTIFACT_ID: {
                    "artifact_id": CHK1_ARTIFACT_ID,
                    "parent_artifact_id": BASE_ARTIFACT_ID,
                    "exact_merge_evidence": str(exact_merge_path),
                    "exact_merge_subject_artifact_id": ("chk1-clean-v2-lr1e6-cp200"),
                },
            },
        },
    )
    _write_sealed(
        semantic_manifest_path,
        {
            "schema_version": "checkpoint-eval-semantic-model-manifest-v1",
            "created_for_evaluation_id": config["evaluation_id"],
            "network_at_scoring_time": False,
            "models": {
                "bertscore": {
                    "local_path": str(bert),
                    "directory_sha256": bert_sha,
                    "resolved_revision": "1" * 40,
                    "num_layers": 17,
                },
                "embedding_cosine": {
                    "local_path": str(mpnet),
                    "directory_sha256": mpnet_sha,
                    "resolved_revision": "2" * 40,
                    "independent_from_training_reward": True,
                },
            },
        },
    )

    generations: list[Path] = []
    generation_manifests: list[Path] = []
    checkpoint_sha = sha256_file(checkpoint_manifest_path)
    config_sha = sha256_file(config_path)
    prompts_sha = sha256_file(prompts_path)
    for artifact_id, model_path, model_sha in (
        (BASE_ARTIFACT_ID, model_base, base_sha),
        (CHK1_ARTIFACT_ID, model_chk1, chk1_sha),
    ):
        generation_path = root / f"{artifact_id}.jsonl"
        generation_manifest_path = root / f"{artifact_id}.manifest.json"
        _write_jsonl(
            generation_path,
            [
                {
                    "artifact_id": artifact_id,
                    "sample_id": row["sample_id"],
                    "meeting_id": row["meeting_id"],
                    "prompt_sha256": row["prompt_sha256"],
                    "generation_model_sha256": model_sha,
                    "generation_tokenizer_sha256": model_sha,
                    "verified_parent_artifact_id": (
                        None if artifact_id == BASE_ARTIFACT_ID else BASE_ARTIFACT_ID
                    ),
                    "test_set_sha256": prompts_sha,
                    "prompt_template_sha256": config_sha,
                    "decoding_config_sha256": config_sha,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 8192,
                    "max_model_len": 24576,
                    "generation_seed_policy": "sample-id-sha256-v1",
                    "generation_seed": derive_row_seed(20260729, row["sample_id"]),
                    "valid_generation": True,
                }
                for row in prompts
            ],
        )
        _write_sealed(
            generation_manifest_path,
            {
                "schema_version": "checkpoint-artifact-generation-manifest-v2",
                "artifact_id": artifact_id,
                "sample_count": 33,
                "valid_count": 33,
                "invalid_count": 0,
                "checkpoint_manifest": {
                    "path": str(checkpoint_manifest_path),
                    "sha256": checkpoint_sha,
                },
                "evaluation_config": {
                    "path": str(config_path),
                    "sha256": sha256_file(config_path),
                },
                "prompts": {
                    "path": str(prompts_path),
                    "sha256": sha256_file(prompts_path),
                    "row_count": 33,
                },
                "model_artifact": {"path": str(model_path), "sha256": model_sha},
                "tokenizer_artifact": {
                    "path": str(model_path),
                    "sha256": model_sha,
                },
                "output": {
                    "path": str(generation_path),
                    "sha256": sha256_file(generation_path),
                    "row_count": 33,
                },
            },
        )
        generations.append(generation_path)
        generation_manifests.append(generation_manifest_path)

    return {
        "generations": generations,
        "generation_manifests": generation_manifests,
        "prompts": prompts_path,
        "references": references_path,
        "config": config_path,
        "test_manifest": test_manifest_path,
        "checkpoint_manifest": checkpoint_manifest_path,
        "lineage_manifest": lineage_manifest_path,
        "semantic_manifest": semantic_manifest_path,
        "exact_merge_evidence": exact_merge_path,
        "repo_root": root,
    }


def _validate(fixture: dict[str, object]):
    return validate_inputs(**fixture)  # type: ignore[arg-type]


def _rebind_exact_evidence(fixture: dict[str, object]) -> None:
    checkpoint_path = fixture["checkpoint_manifest"]
    evidence_path = fixture["exact_merge_evidence"]
    assert isinstance(checkpoint_path, Path)
    assert isinstance(evidence_path, Path)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint.pop("integrity")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    checkpoint["bindings"]["exact_merge_evidence"].update(
        {
            "sha256": sha256_file(evidence_path),
            "payload_sha256": evidence["integrity"]["payload_sha256"],
        }
    )
    _write_sealed(checkpoint_path, checkpoint)

    checkpoint_sha = sha256_file(checkpoint_path)
    for manifest_path in fixture["generation_manifests"]:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("integrity")
        manifest["checkpoint_manifest"]["sha256"] = checkpoint_sha
        _write_sealed(manifest_path, manifest)


def test_validates_complete_two_artifact_matrix(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    _, _, _, receipt = _validate(fixture)

    assert receipt["status"] == "validated"
    assert receipt["complete"] is True
    assert receipt["artifact_ids"] == list(ARTIFACT_IDS)
    assert receipt["frozen_test"]["sample_count"] == 33
    assert receipt["exact_merge_evidence"]["tensor_verification"]["complete"] is True
    assert receipt["core_evaluator"] == {
        "module": "jobs.eval.eval_checkpoint_generation",
        "expected_artifact_count": 2,
        "paired_unit": "sample_id",
        "cluster_unit": "meeting_id",
        "multiple_testing": "Holm",
    }


def test_rejects_missing_generation_manifest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["generation_manifests"] = fixture["generation_manifests"][:1]

    with pytest.raises(ValueError, match="Exactly 2 generation manifests"):
        _validate(fixture)


def test_rejects_frozen_reference_mutation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    references = fixture["references"]
    assert isinstance(references, Path)
    references.write_text(
        references.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Frozen references"):
        _validate(fixture)


def test_rejects_exact_merge_model_hash_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence_path = fixture["exact_merge_evidence"]
    assert isinstance(evidence_path, Path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence.pop("integrity")
    evidence["sources"]["merged_model"]["sha256"] = "f" * 64
    _write_sealed(evidence_path, evidence)
    _rebind_exact_evidence(fixture)

    with pytest.raises(ValueError, match="merged_model hash differs"):
        _validate(fixture)


def test_rejects_incomplete_exact_tensor_inventory(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence_path = fixture["exact_merge_evidence"]
    assert isinstance(evidence_path, Path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence.pop("integrity")
    evidence["tensor_verification"]["exact_adapted_model_tensor_count"] = 223
    _write_sealed(evidence_path, evidence)
    _rebind_exact_evidence(fixture)

    with pytest.raises(ValueError, match="every model tensor exactly"):
        _validate(fixture)


def test_rejects_generation_model_not_bound_to_checkpoint(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = fixture["generation_manifests"][1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("integrity")
    manifest["model_artifact"]["sha256"] = "f" * 64
    _write_sealed(manifest_path, manifest)

    with pytest.raises(ValueError, match="model_artifact differs"):
        _validate(fixture)


def test_rejects_failed_progress_chain_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)

    def fail_closed(**_kwargs):
        raise ValueError("frozen progress state binding changed")

    monkeypatch.setattr(
        "jobs.eval.eval_retrain_v2_chk0_chk1_generation."
        "validate_generation_progress_binding",
        fail_closed,
    )
    with pytest.raises(ValueError, match="frozen progress state binding changed"):
        _validate(fixture)


@pytest.mark.parametrize(
    "policy",
    ("strict-final-answer-v2", "length-tolerant-open-tags-v1"),
)
def test_cli_supports_both_scoring_policies(policy: str) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--generations",
            "a.jsonl",
            "--generations",
            "b.jsonl",
            "--generation-manifest",
            "a.manifest.json",
            "--generation-manifest",
            "b.manifest.json",
            "--prompts",
            "prompts.jsonl",
            "--references",
            "references.jsonl",
            "--config",
            "config.json",
            "--test-manifest",
            "test.json",
            "--checkpoint-manifest",
            "checkpoint.json",
            "--lineage-manifest",
            "lineage.json",
            "--semantic-manifest",
            "semantic.json",
            "--exact-merge-evidence",
            "evidence.json",
            "--output-dir",
            "scores",
            "--scoring-policy",
            policy,
        ]
    )

    assert args.scoring_policy == policy


def test_versioned_runner_does_not_target_historical_output() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "run/eval_retrain_v2_chk0_chk1_generation.sh").read_text(
        encoding="utf-8"
    )

    assert "retrain_v2_chk0_chk1_cp200_20260810_v1" in runner
    assert "eval-chk0-base" in runner
    assert "eval-chk1-clean-v2-lr1e6-cp200" in runner
    assert "eval_retrain_v2_chk0_chk1_generation" in runner
    assert "score-length-tolerant" in runner
    assert "summarize OUTPUT_PATH" in runner
    assert "jobs.eval.summarize_chk0_chk1_cp200_generation_eval" in runner
    assert '--strict-dir "${SCORE_DIR}"' in runner
    assert '--length-tolerant-dir "${LENGTH_TOLERANT_SCORE_DIR}"' in runner
    assert "retrain_v2_chk0_chk1_chk2_cp150_20260809" not in runner
