from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jobs.eval import eval_retrain_v2_chk0_chk1_chk3_generation as subject
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import derive_row_seed, seal_manifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_sealed(path: Path, payload: dict) -> dict:
    sealed = seal_manifest(payload)
    _write_json(path, sealed)
    return sealed


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _binding(path: Path, payload: dict | None = None) -> dict:
    result = {"path": str(path), "file_sha256": sha256_file(path)}
    if payload is not None:
        result["payload_sha256"] = payload["integrity"]["payload_sha256"]
    return result


def _dir(root: Path, name: str) -> Path:
    result = root / name
    result.mkdir()
    (result / "marker").write_text(name, encoding="utf-8")
    return result


@pytest.fixture(autouse=True)
def _stub_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subject,
        "validate_generation_progress_binding",
        lambda **_kwargs: {
            "contract_sha256": "1" * 64,
            "state_sha256": "2" * 64,
            "partial_sha256": "3" * 64,
            "row_count": 33,
        },
    )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    root = tmp_path.resolve()
    base_model = _dir(root, "base")
    chk1_model = _dir(root, "chk1")
    chk3_model = _dir(root, "chk3")
    chk3_tokenizer = chk3_model
    chk3_adapter = _dir(root, "adapter-cp250")
    semantic_bert = _dir(root, "bert")
    semantic_mpnet = _dir(root, "mpnet")
    base_sha = "a" * 64
    chk1_sha = "b" * 64
    chk3_sha = "c" * 64
    adapter_sha = "d" * 64
    adapter_weights_sha = "e" * 64

    config_path = root / "config.json"
    config = {
        "schema_version": "checkpoint-generation-eval-config-v1",
        "evaluation_id": "frozen-three-leg-test",
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 8192,
            "max_model_len": 24576,
            "seed_policy": "sample-id-sha256-v1",
            "base_seed": 20260729,
        },
    }
    _write_json(config_path, config)
    prompts_path = root / "prompts.jsonl"
    references_path = root / "references.jsonl"
    prompt_rows = []
    for index in range(33):
        prompt = f"prompt-{index}"
        prompt_rows.append(
            {
                "sample_id": f"sample-{index:02d}",
                "meeting_id": f"meeting-{index // 3:02d}",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    _write_jsonl(prompts_path, prompt_rows)
    _write_jsonl(
        references_path,
        [{"sample_id": row["sample_id"], "reference": "reference"} for row in prompt_rows],
    )
    test_manifest_path = root / "test-manifest.json"
    test_manifest = _write_sealed(
        test_manifest_path,
        {
            "schema_version": "checkpoint-eval-test-manifest-v1",
            "evaluation_id": config["evaluation_id"],
            "sample_count": 33,
            "section_count": 3,
            "meeting_dates": [f"meeting-{index:02d}" for index in range(11)],
            "prospective_only_meeting_dates": [
                f"meeting-{index:02d}" for index in range(2, 11)
            ],
            "reference_in_prompt": False,
            "secondary_llm_summary": False,
        },
    )
    semantic_path = root / "semantic.json"
    semantic = _write_sealed(
        semantic_path,
        {
            "schema_version": "checkpoint-eval-semantic-model-manifest-v1",
            "created_for_evaluation_id": config["evaluation_id"],
            "network_at_scoring_time": False,
            "models": {
                "bertscore": {
                    "local_path": str(semantic_bert),
                    "directory_sha256": "4" * 64,
                    "resolved_revision": "4" * 40,
                    "num_layers": 17,
                },
                "embedding_cosine": {
                    "local_path": str(semantic_mpnet),
                    "directory_sha256": "5" * 64,
                    "resolved_revision": "5" * 40,
                    "independent_from_training_reward": True,
                },
            },
        },
    )

    pair_checkpoint_path = root / "pair-checkpoint.json"
    pair_checkpoint = _write_sealed(
        pair_checkpoint_path,
        {
            "schema_version": "retrain-v2-checkpoint-eval-manifest-v1",
            "artifacts": [
                {
                    "artifact_id": subject.BASE_ARTIFACT_ID,
                    "design_checkpoint_id": "chk-0",
                    "intended_parent_id": None,
                    "verified_parent_artifact_id": None,
                    "model_path": str(base_model),
                    "model_sha256": base_sha,
                    "tokenizer_path": str(base_model),
                    "tokenizer_sha256": base_sha,
                    "usable_for_evaluation": True,
                },
                {
                    "artifact_id": subject.CHK1_ARTIFACT_ID,
                    "design_checkpoint_id": "chk-1",
                    "intended_parent_id": "chk-0",
                    "verified_parent_artifact_id": subject.BASE_ARTIFACT_ID,
                    "model_path": str(chk1_model),
                    "model_sha256": chk1_sha,
                    "tokenizer_path": str(chk1_model),
                    "tokenizer_sha256": chk1_sha,
                    "usable_for_evaluation": True,
                },
            ],
        },
    )
    pair_exact_path = root / "pair-exact.json"
    pair_exact = _write_sealed(
        pair_exact_path,
        {
            "schema_version": "lora-merge-lineage-evidence-v1",
            "subject_artifact_id": "chk1-clean-v2-lr1e6-cp200",
        },
    )
    pair_lineage_path = root / "pair-lineage.json"
    pair_lineage = _write_sealed(
        pair_lineage_path,
        {
            "schema_version": "retrain-v2-eval-lineage-v1",
            "checkpoints": {
                subject.BASE_ARTIFACT_ID: {
                    "artifact_id": subject.BASE_ARTIFACT_ID,
                    "parent_artifact_id": None,
                },
                subject.CHK1_ARTIFACT_ID: {
                    "artifact_id": subject.CHK1_ARTIFACT_ID,
                    "parent_artifact_id": subject.BASE_ARTIFACT_ID,
                    "exact_merge_evidence": _binding(pair_exact_path, pair_exact),
                    "exact_merge_subject_artifact_id": "chk1-clean-v2-lr1e6-cp200",
                },
            },
        },
    )

    selection_path = root / "selection.json"
    _write_json(
        selection_path,
        {
            "schema_version": "chk3-standalone-checkpoint-selection-v1",
            "status": "selected",
            "selected_checkpoint": 250,
            "scope": {
                "stage": "chk3",
                "promotable_to_canonical_dag": False,
            },
            "checkpoint": {
                "path": str(chk3_adapter),
                "adapter_model_sha256": adapter_weights_sha,
                "adapter_tensor_count": 448,
                "nonfinite_tensor_count": 0,
            },
            "generation_gate": {"status": "passed", "cases": 3, "quality_valid_cases": 3},
        },
    )
    authorization_path = root / "authorization.json"
    _write_json(
        authorization_path,
        {
            "schema_version": "chk3-cp250-generation-comparison-authorization-v1",
            "status": "authorized",
            "scope": {
                "checkpoint_step": 250,
                "artifact_status": "evaluation_only",
                "canonical_dag_promotable": False,
                "downstream_training_allowed": False,
            },
            "bindings": {
                "base_model": {"path": str(chk1_model), "sha256": chk1_sha},
                "adapter_checkpoint": {
                    "path": str(chk3_adapter),
                    "directory_sha256": adapter_sha,
                },
                "merged_destination": str(chk3_model),
            },
        },
    )
    chk3_exact_path = root / "chk3-exact.json"
    chk3_exact = _write_sealed(
        chk3_exact_path,
        {
            "schema_version": "lora-merge-lineage-evidence-v1",
            "algorithm_version": "peft-lora-fp32-exact-v1",
            "subject_artifact_id": subject.CHK3_ARTIFACT_ID,
            "conclusion": "exact_base_plus_adapter_merge_verified",
            "sources": {
                "base_model": {"path": str(chk1_model), "sha256": chk1_sha},
                "merged_model": {"path": str(chk3_model), "sha256": chk3_sha},
                "adapter": {"path": str(chk3_adapter), "sha256": adapter_sha},
                "critical_files": {
                    "adapter_weights": {"sha256": adapter_weights_sha}
                },
            },
            "metadata_evidence": {
                "adapter": {"base_path_matches": True},
                "training_config": {
                    "lora_metadata_checks": {"r_matches": True},
                    "path_checks": {"base_matches": True},
                },
            },
            "tensor_verification": {
                "model_tensor_count": 2,
                "adapted_model_tensor_count": 1,
                "exact_adapted_model_tensor_count": 1,
                "unchanged_model_tensor_count": 1,
                "exact_unchanged_model_tensor_count": 1,
                "adapter_tensor_count": 2,
                "mismatch_count": 0,
                "adapted_tensors": [
                    {"tensor_name": "layer.weight", "exact_reconstruction": True}
                ],
            },
        },
    )
    chk3_checkpoint_path = root / "chk3-checkpoint.json"
    chk3_checkpoint = _write_sealed(
        chk3_checkpoint_path,
        {
            "schema_version": "retrain-v2-checkpoint-eval-manifest-v1",
            "evaluation_only": True,
            "promotable_to_canonical_dag": False,
            "artifacts": [
                {
                    "artifact_id": subject.CHK3_ARTIFACT_ID,
                    "design_checkpoint_id": "chk-3-direct-cp250",
                    "intended_parent_id": "chk-1",
                    "verified_parent_artifact_id": subject.CHK1_ARTIFACT_ID,
                    "model_path": str(chk3_model),
                    "model_sha256": chk3_sha,
                    "tokenizer_path": str(chk3_tokenizer),
                    "tokenizer_sha256": chk3_sha,
                    "adapter_path": str(chk3_adapter),
                    "adapter_sha256": adapter_sha,
                    "usable_for_evaluation": True,
                }
            ],
            "bindings": {
                "evaluation_config": _binding(config_path),
                "test_manifest": _binding(test_manifest_path, test_manifest),
                "parent_checkpoint_manifest": _binding(
                    pair_checkpoint_path, pair_checkpoint
                ),
                "selection_receipt": _binding(selection_path),
                "authorization": _binding(authorization_path),
                "exact_merge_evidence": _binding(chk3_exact_path, chk3_exact),
            },
        },
    )
    three_lineage_path = root / "three-lineage.json"
    _write_sealed(
        three_lineage_path,
        {
            "schema_version": "retrain-v2-eval-lineage-v1",
            "bindings": {
                "parent_checkpoint_manifest": _binding(
                    pair_checkpoint_path, pair_checkpoint
                ),
                "chk3_checkpoint_manifest": _binding(
                    chk3_checkpoint_path, chk3_checkpoint
                ),
                "parent_lineage_manifest": _binding(pair_lineage_path, pair_lineage),
            },
            "checkpoints": {
                subject.BASE_ARTIFACT_ID: {
                    "artifact_id": subject.BASE_ARTIFACT_ID,
                    "parent_artifact_id": None,
                },
                subject.CHK1_ARTIFACT_ID: {
                    "artifact_id": subject.CHK1_ARTIFACT_ID,
                    "parent_artifact_id": subject.BASE_ARTIFACT_ID,
                    "exact_merge_evidence": _binding(pair_exact_path, pair_exact),
                    "exact_merge_subject_artifact_id": "chk1-clean-v2-lr1e6-cp200",
                },
                subject.CHK3_ARTIFACT_ID: {
                    "artifact_id": subject.CHK3_ARTIFACT_ID,
                    "parent_artifact_id": subject.CHK1_ARTIFACT_ID,
                    "exact_merge_evidence": _binding(chk3_exact_path, chk3_exact),
                    "exact_merge_subject_artifact_id": subject.CHK3_ARTIFACT_ID,
                },
            },
        },
    )

    generations: list[Path] = []
    generation_manifests: list[Path] = []
    for artifact_id in subject.PAIR_ARTIFACT_IDS:
        generation = root / f"{artifact_id}.jsonl"
        _write_jsonl(generation, [{"artifact_id": artifact_id}])
        manifest_path = root / f"{artifact_id}.manifest.json"
        _write_sealed(
            manifest_path,
            {
                "schema_version": "checkpoint-artifact-generation-manifest-v2",
                "artifact_id": artifact_id,
            },
        )
        generations.append(generation)
        generation_manifests.append(manifest_path)

    chk3_generation_path = root / f"{subject.CHK3_ARTIFACT_ID}.jsonl"
    config_sha = sha256_file(config_path)
    prompts_sha = sha256_file(prompts_path)
    chk3_rows = [
        {
            "artifact_id": subject.CHK3_ARTIFACT_ID,
            "sample_id": row["sample_id"],
            "meeting_id": row["meeting_id"],
            "prompt_sha256": row["prompt_sha256"],
            "generation_model_sha256": chk3_sha,
            "generation_tokenizer_sha256": chk3_sha,
            "verified_parent_artifact_id": subject.CHK1_ARTIFACT_ID,
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
        for row in prompt_rows
    ]
    _write_jsonl(chk3_generation_path, chk3_rows)
    chk3_generation_manifest_path = root / f"{subject.CHK3_ARTIFACT_ID}.manifest.json"
    _write_sealed(
        chk3_generation_manifest_path,
        {
            "schema_version": "checkpoint-artifact-generation-manifest-v2",
            "artifact_id": subject.CHK3_ARTIFACT_ID,
            "sample_count": 33,
            "valid_count": 33,
            "invalid_count": 0,
            "checkpoint_manifest": {
                "path": str(chk3_checkpoint_path),
                "sha256": sha256_file(chk3_checkpoint_path),
            },
            "prompts": {
                "path": str(prompts_path),
                "sha256": prompts_sha,
                "row_count": 33,
            },
            "evaluation_config": {
                "path": str(config_path),
                "sha256": config_sha,
            },
            "model_artifact": {"path": str(chk3_model), "sha256": chk3_sha},
            "tokenizer_artifact": {"path": str(chk3_model), "sha256": chk3_sha},
            "output": {
                "path": str(chk3_generation_path),
                "sha256": sha256_file(chk3_generation_path),
                "row_count": 33,
            },
        },
    )
    generations.append(chk3_generation_path)
    generation_manifests.append(chk3_generation_manifest_path)

    pair_validation = seal_manifest(
        {
            "status": "validated",
            "complete": True,
            "frozen_test": {
                "sample_count": 33,
                "prompts": {"path": str(prompts_path), "sha256": prompts_sha},
            },
            "checkpoint_manifest": _binding(pair_checkpoint_path, pair_checkpoint),
            "lineage_manifest": _binding(pair_lineage_path, pair_lineage),
            "exact_merge_evidence": _binding(pair_exact_path, pair_exact),
            "semantic_manifest": _binding(semantic_path, semantic),
            "generations": {
                artifact_id: {"sample_count": 33, "progress": {"row_count": 33}}
                for artifact_id in subject.PAIR_ARTIFACT_IDS
            },
        }
    )

    def validate_pair(**kwargs):
        manifests = kwargs["generation_manifests"]
        assert len(manifests) == 2
        observed = []
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert manifest["schema_version"] == "checkpoint-artifact-generation-manifest-v2"
            observed.append(manifest["artifact_id"])
        assert observed == list(subject.PAIR_ARTIFACT_IDS)
        return config, test_manifest, semantic, pair_validation

    monkeypatch.setattr(subject.pair_eval, "validate_inputs", validate_pair)
    return {
        "generations": generations,
        "generation_manifests": generation_manifests,
        "prompts": prompts_path,
        "references": references_path,
        "config": config_path,
        "test_manifest": test_manifest_path,
        "pair_checkpoint_manifest": pair_checkpoint_path,
        "pair_lineage_manifest": pair_lineage_path,
        "semantic_manifest": semantic_path,
        "pair_exact_merge_evidence": pair_exact_path,
        "chk3_checkpoint_manifest": chk3_checkpoint_path,
        "three_leg_lineage_manifest": three_lineage_path,
        "repo_root": root,
    }


def _validate(fixture: dict[str, object]):
    return subject.validate_inputs(**fixture)  # type: ignore[arg-type]


def test_validates_complete_three_leg_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    _, _, _, receipt, ordered = _validate(fixture)

    assert receipt["status"] == "validated"
    assert receipt["complete"] is True
    assert receipt["artifact_ids"] == list(subject.ARTIFACT_IDS)
    assert receipt["evaluation_only"] is True
    assert receipt["promotable_to_canonical_dag"] is False
    assert receipt["core_evaluator"]["expected_artifact_count"] == 3
    assert receipt["core_evaluator"]["required_artifact_ids"] == list(subject.ARTIFACT_IDS)
    assert receipt["three_leg_lineage"]["contrasts"] == [
        {"baseline": subject.BASE_ARTIFACT_ID, "candidate": subject.CHK1_ARTIFACT_ID},
        {"baseline": subject.CHK1_ARTIFACT_ID, "candidate": subject.CHK3_ARTIFACT_ID},
    ]
    assert [path.name.removesuffix(".jsonl") for path in ordered] == list(subject.ARTIFACT_IDS)


def test_fails_closed_when_original_pair_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    def fail_pair(**_kwargs):
        raise ValueError("sealed pair progress chain changed")

    monkeypatch.setattr(subject.pair_eval, "validate_inputs", fail_pair)
    with pytest.raises(ValueError, match="sealed pair progress chain changed"):
        _validate(fixture)


def test_rejects_tampered_chk3_generation_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    generation = fixture["generations"][-1]  # type: ignore[index]
    assert isinstance(generation, Path)
    generation.write_text(generation.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="chk3 generation output binding changed"):
        _validate(fixture)


def test_rejects_tampered_chk3_decoding_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    generation = fixture["generations"][-1]  # type: ignore[index]
    manifest_path = fixture["generation_manifests"][-1]  # type: ignore[index]
    assert isinstance(generation, Path) and isinstance(manifest_path, Path)
    rows = [json.loads(line) for line in generation.read_text().splitlines()]
    rows[0]["temperature"] = 0.6
    _write_jsonl(generation, rows)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("integrity")
    manifest["output"]["sha256"] = sha256_file(generation)
    _write_sealed(manifest_path, manifest)

    with pytest.raises(ValueError, match="generation row provenance changed"):
        _validate(fixture)


def test_rejects_tampered_exact_merge_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    checkpoint_path = fixture["chk3_checkpoint_manifest"]
    assert isinstance(checkpoint_path, Path)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    evidence_path = Path(checkpoint["bindings"]["exact_merge_evidence"]["path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence.pop("integrity")
    evidence["sources"]["merged_model"]["sha256"] = "f" * 64
    evidence = _write_sealed(evidence_path, evidence)
    checkpoint.pop("integrity")
    checkpoint["bindings"]["exact_merge_evidence"] = _binding(evidence_path, evidence)
    _write_sealed(checkpoint_path, checkpoint)

    with pytest.raises(ValueError, match="merged_model differs"):
        _validate(fixture)


def test_rejects_nonsealed_chk3_design_checkpoint_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    checkpoint_path = fixture["chk3_checkpoint_manifest"]
    assert isinstance(checkpoint_path, Path)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint.pop("integrity")
    checkpoint["artifacts"][0]["design_checkpoint_id"] = "chk-3"
    _write_sealed(checkpoint_path, checkpoint)

    with pytest.raises(ValueError, match="parent/design/evaluation contract changed"):
        _validate(fixture)


def test_rejects_tampered_three_leg_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    lineage_path = fixture["three_leg_lineage_manifest"]
    assert isinstance(lineage_path, Path)
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage.pop("integrity")
    lineage["checkpoints"][subject.CHK3_ARTIFACT_ID]["parent_artifact_id"] = subject.BASE_ARTIFACT_ID
    _write_sealed(lineage_path, lineage)

    with pytest.raises(ValueError, match="lineage parent changed"):
        _validate(fixture)


@pytest.mark.parametrize(
    "policy", ("strict-final-answer-v2", "length-tolerant-open-tags-v1")
)
def test_cli_supports_strict_and_length_tolerant(policy: str) -> None:
    args = subject.build_parser().parse_args(
        [
            "--generations", "a.jsonl",
            "--generations", "b.jsonl",
            "--generations", "c.jsonl",
            "--generation-manifest", "a.manifest.json",
            "--generation-manifest", "b.manifest.json",
            "--generation-manifest", "c.manifest.json",
            "--prompts", "prompts.jsonl",
            "--references", "references.jsonl",
            "--config", "config.json",
            "--test-manifest", "test.json",
            "--pair-checkpoint-manifest", "pair-checkpoint.json",
            "--pair-lineage-manifest", "pair-lineage.json",
            "--semantic-manifest", "semantic.json",
            "--pair-exact-merge-evidence", "pair-exact.json",
            "--chk3-checkpoint-manifest", "chk3-checkpoint.json",
            "--three-leg-lineage-manifest", "three-lineage.json",
            "--output-dir", "scores",
            "--scoring-policy", policy,
        ]
    )
    assert args.scoring_policy == policy


def test_main_calls_core_with_exact_three_leg_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _dir(tmp_path, "semantic-model")
    generation_paths = [tmp_path / f"{artifact}.jsonl" for artifact in subject.ARTIFACT_IDS]
    validation = seal_manifest({"status": "validated"})
    semantic = {
        "models": {
            "bertscore": {
                "local_path": str(model),
                "directory_sha256": "1" * 64,
                "num_layers": 1,
            },
            "embedding_cosine": {
                "local_path": str(model),
                "directory_sha256": "2" * 64,
            },
        }
    }
    monkeypatch.setattr(
        subject,
        "validate_inputs",
        lambda **_kwargs: (
            {},
            {"prospective_only_meeting_dates": ["meeting-02"]},
            semantic,
            validation,
            generation_paths,
        ),
    )
    monkeypatch.setattr(subject, "BERTScoreBackend", lambda *_args, **_kwargs: "bert")
    monkeypatch.setattr(subject, "MPNetCosineBackend", lambda *_args, **_kwargs: "mpnet")
    observed: dict = {}

    def fake_core(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {"audit": {"status": "complete"}, "reused_existing": False, "paths": {}}

    monkeypatch.setattr(subject, "run_checkpoint_generation_evaluation", fake_core)
    monkeypatch.setattr(
        subject,
        "_write_validation_receipt",
        lambda **_kwargs: tmp_path / "input_validation.json",
    )
    args = [
        item
        for artifact in ("a", "b", "c")
        for item in ("--generations", f"{artifact}.jsonl")
    ] + [
        item
        for artifact in ("a", "b", "c")
        for item in ("--generation-manifest", f"{artifact}.manifest.json")
    ] + [
        "--prompts", "prompts.jsonl",
        "--references", "references.jsonl",
        "--config", "config.json",
        "--test-manifest", "test.json",
        "--pair-checkpoint-manifest", "pair-checkpoint.json",
        "--pair-lineage-manifest", "pair-lineage.json",
        "--semantic-manifest", "semantic.json",
        "--pair-exact-merge-evidence", "pair-exact.json",
        "--chk3-checkpoint-manifest", "chk3-checkpoint.json",
        "--three-leg-lineage-manifest", "three-lineage.json",
        "--output-dir", str(tmp_path / "scores"),
        "--repo-root", str(tmp_path),
    ]
    assert subject.main(args) == 0
    assert observed["args"][0] == generation_paths
    assert observed["kwargs"]["require_formal_manifests"] is False
    assert observed["kwargs"]["expected_artifact_count"] == 3
    assert observed["kwargs"]["required_artifact_ids"] == subject.ARTIFACT_IDS
    assert observed["kwargs"]["lineage_manifest_path"] == Path("three-lineage.json")


def test_runner_scores_only_and_uses_gpu0_three_leg_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "run/eval_retrain_v2_chk0_chk1_chk3_generation.sh").read_text(
        encoding="utf-8"
    )
    assert "score-length-tolerant" in runner
    assert "strict-final-answer-v2" in runner
    assert "length-tolerant-open-tags-v1" in runner
    assert "eval-chk0-base" in runner
    assert "eval-chk1-clean-v2-lr1e6-cp200" in runner
    assert subject.CHK3_ARTIFACT_ID in runner
    assert "CHECKPOINT_EVAL_CHK3_CHECKPOINT_MANIFEST" in runner
    assert "CHECKPOINT_EVAL_THREE_LEG_LINEAGE_MANIFEST" in runner
    assert "CHECKPOINT_EVAL_SEMANTIC_GPU_INDEX:-0" in runner
    assert "generate)" not in runner
