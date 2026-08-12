import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jobs.eval.eval_checkpoint_generation import (
    BERTScoreBackend,
    FORMAL_PROSPECTIVE_SAMPLE_COUNT,
    FORMAL_TEST_SAMPLE_COUNT,
    LENGTH_TOLERANT_SCORING_POLICY,
    MPNetCosineBackend,
    _sha256_path,
    run_checkpoint_generation_evaluation,
)
from open_r1.validator.loo_generation_spec import derive_row_seed, seal_manifest
from open_r1.validator.text_leakage import (
    PROMPT_REFERENCE_OVERLAP_TOKEN_COUNT,
    find_prompt_reference_token_overlap,
)


MEETINGS = [
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-10",
    "2026-01-28",
    "2026-03-18",
    "2026-04-29",
    "2026-06-17",
]
PROSPECTIVE = MEETINGS[2:]
SECTIONS = ["participants_views", "economic_situation", "financial_situation"]
ARTIFACT_PARENTS = {
    "eval-base": None,
    "eval-analysis-sft": "eval-base",
    "eval-legacy-grpo-from-chk0": "eval-base",
    "eval-minutes-sft-from-chk1": "eval-analysis-sft",
}


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


class FakeBERTScoreBackend(BERTScoreBackend):
    def __init__(self, model_path: Path, model_sha256: str) -> None:
        self.path = model_path.resolve()
        self.model_id = str(self.path)
        self.model_sha256 = model_sha256
        self.num_layers = 17
        self.batch_size = 4
        self.device = "cpu"
        self.max_length = 512
        self._chunk_audit = None
        self.candidate_count = 0

    def score(self, candidates, references):
        self.candidate_count += len(candidates)
        return {
            metric: [
                1.0 if candidate == reference else 0.5
                for candidate, reference in zip(
                    candidates,
                    references,
                    strict=True,
                )
            ]
            for metric in (
                "bertscore_precision",
                "bertscore_recall",
                "bertscore_f1",
            )
        }


class FakeMPNetBackend(MPNetCosineBackend):
    def __init__(self, model_path: Path, model_sha256: str) -> None:
        self.path = model_path.resolve()
        self.model_id = str(self.path)
        self.model_sha256 = model_sha256
        self.batch_size = 4
        self.device = "cpu"
        self.max_length = 512
        self._chunk_audit = None
        self.candidate_count = 0

    def score(self, candidates, references):
        self.candidate_count += len(candidates)
        return [
            1.0 if candidate == reference else 0.5
            for candidate, reference in zip(candidates, references, strict=True)
        ]


def _build_formal_fixture(root: Path) -> dict:
    config_path = root / "config.json"
    prompts_path = root / "prompts.jsonl"
    references_path = root / "references.jsonl"
    test_manifest_path = root / "test_manifest.json"
    checkpoint_manifest_path = root / "checkpoint_manifest.json"
    semantic_manifest_path = root / "semantic_manifest.json"
    lineage_evidence_path = root / "lineage_evidence.json"
    config = {
        "schema_version": "checkpoint-generation-eval-config-v1",
        "evaluation_id": "fixture-evaluation",
        "population": {
            "meeting_dates": MEETINGS,
            "prospective_only_meeting_dates": PROSPECTIVE,
        },
        "sections": [
            {"section_id": section_id, "section_name": section_id}
            for section_id in SECTIONS
        ],
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "base_seed": 20260729,
            "seed_policy": "sample-id-sha256-v1",
            "max_new_tokens": 64,
            "max_model_len": 512,
        },
        "evaluation": {
            "bootstrap_iterations": 10_000,
            "bootstrap_seed": 20260729,
            "score_final_answer_only": True,
            "long_text_policy": "sentence-boundary-chunk-weighted-v1",
        },
    }
    _write_json(config_path, config)

    prompts = []
    references = []
    for meeting_date in MEETINGS:
        for section_id in SECTIONS:
            sample_id = f"{meeting_date}::{section_id}"
            prompt = f"Evidence for {sample_id}"
            reference = "Inflation declined while employment increased."
            facts = [
                {
                    "indicator": "Inflation",
                    "series_id": "PCE",
                    "kind": "derived_absolute_change",
                    "from_date": "2025-01-01",
                    "to_date": "2025-02-01",
                    "value": "-0.1",
                    "unit": "Percent",
                },
                {
                    "indicator": "Employment",
                    "series_id": "PAYEMS",
                    "kind": "derived_absolute_change",
                    "from_date": "2025-01-01",
                    "to_date": "2025-02-01",
                    "value": "1",
                    "unit": "Thousands",
                },
                {
                    "indicator": "Federal-Funds-Rate",
                    "series_id": "DFF",
                    "kind": "derived_absolute_change",
                    "from_date": "2025-01-01",
                    "to_date": "2025-02-01",
                    "value": "0",
                    "unit": "Percentage Points",
                },
            ]
            evidence_sha256 = _sha_text(json.dumps(facts, sort_keys=True))
            prompts.append(
                {
                    "sample_id": sample_id,
                    "meeting_date": meeting_date,
                    "section_id": section_id,
                    "section_name": section_id,
                    "prompt": prompt,
                    "prompt_sha256": _sha_text(prompt),
                    "evidence_sha256": evidence_sha256,
                    "evidence_facts": facts,
                    "reference_used_in_prompt": False,
                }
            )
            references.append(
                {
                    "sample_id": sample_id,
                    "meeting_date": meeting_date,
                    "section_id": section_id,
                    "section_name": section_id,
                    "reference": reference,
                    "reference_sha256": _sha_text(reference),
                }
            )
    _write_jsonl(prompts_path, prompts)
    _write_jsonl(references_path, references)

    test_manifest = seal_manifest(
        {
            "schema_version": "checkpoint-eval-test-manifest-v1",
            "evaluation_id": "fixture-evaluation",
            "meeting_dates": MEETINGS,
            "prospective_only_meeting_dates": PROSPECTIVE,
            "section_count": 3,
            "sample_count": 33,
            "reference_in_prompt": False,
            "inputs": {
                "config": {
                    "path": str(config_path),
                    "sha256": _sha_file(config_path),
                }
            },
            "outputs": {
                "prompts": {
                    "path": str(prompts_path),
                    "sha256": _sha_file(prompts_path),
                    "row_count": 33,
                },
                "references": {
                    "path": str(references_path),
                    "sha256": _sha_file(references_path),
                    "row_count": 33,
                },
            },
        }
    )
    _write_json(test_manifest_path, test_manifest)

    artifact_hashes = {
        artifact_id: (f"{index:x}" * 64, f"{index:x}" * 64)
        for index, artifact_id in enumerate(ARTIFACT_PARENTS, start=1)
    }
    checkpoint_manifest = seal_manifest(
        {
            "schema_version": "checkpoint-provenance-manifest-v1",
            "artifacts": [
                {
                    "artifact_id": artifact_id,
                    "verified_parent_artifact_id": parent,
                    "model_sha256": artifact_hashes[artifact_id][0],
                    "model_path": str(root / f"{artifact_id}-model"),
                    "tokenizer_sha256": artifact_hashes[artifact_id][1],
                    "tokenizer_path": str(root / f"{artifact_id}-model"),
                    "usable_for_evaluation": True,
                }
                for artifact_id, parent in ARTIFACT_PARENTS.items()
            ],
        }
    )
    _write_json(checkpoint_manifest_path, checkpoint_manifest)
    lineage_evidence = seal_manifest(
        {
            "schema_version": "lora-merge-lineage-evidence-v1",
            "algorithm_version": "peft-lora-fp32-exact-v1",
            "conclusion": "exact_base_plus_adapter_merge_verified",
            "subject_artifact_id": "eval-analysis-sft",
            "bindings": {
                "checkpoint_manifest": {
                    "path": str(checkpoint_manifest_path),
                    "sha256": _sha_file(checkpoint_manifest_path),
                    "payload_sha256": checkpoint_manifest["integrity"][
                        "payload_sha256"
                    ],
                    "base_artifact_id": "eval-base",
                    "merged_artifact_id": "eval-analysis-sft",
                    "assertions": {
                        "base_fingerprint_matches": True,
                        "base_path_matches": True,
                        "merged_fingerprint_matches": True,
                        "merged_path_matches": True,
                        "merged_verified_parent_matches_base": True,
                    },
                }
            },
            "sources": {
                "base_model": {
                    "path": str(root / "eval-base-model"),
                    "sha256": artifact_hashes["eval-base"][0],
                },
                "merged_model": {
                    "path": str(root / "eval-analysis-sft-model"),
                    "sha256": artifact_hashes["eval-analysis-sft"][0],
                },
            },
            "metadata_evidence": {
                "adapter": {"base_path_matches": True},
                "training_config": {
                    "lora_metadata_checks": {
                        "bias_matches": True,
                        "lora_alpha_matches": True,
                        "r_matches": True,
                        "target_modules_match": True,
                    },
                    "path_checks": {
                        "model_name_or_path_matches_base": True,
                        "output_dir_matches_adapter": True,
                        "peft_merged_model_path_matches_merged": True,
                    },
                },
            },
            "tensor_verification": {
                "adapted_model_tensor_count": 1,
                "exact_adapted_model_tensor_count": 1,
                "unchanged_model_tensor_count": 1,
                "exact_unchanged_model_tensor_count": 1,
                "model_tensor_count": 2,
                "adapter_tensor_count": 2,
                "mismatch_count": 0,
                "total_adapted_elements": 4,
                "total_unchanged_elements": 4,
                "changed_adapted_elements_vs_base": 2,
                "comparison": "torch.equal (exact stored tensor values)",
                "merge_expression": "base.float32 + B.float32 @ A.float32",
                "adapted_tensors": [
                    {
                        "tensor_name": "layer.q_proj.weight",
                        "exact_reconstruction": True,
                    }
                ],
            },
        }
    )
    _write_json(lineage_evidence_path, lineage_evidence)

    generation_paths = []
    generation_manifest_paths = []
    invalid_key = (
        "eval-minutes-sft-from-chk1",
        f"{MEETINGS[-1]}::{SECTIONS[-1]}",
    )
    for artifact_id, parent in ARTIFACT_PARENTS.items():
        generation_path = root / f"{artifact_id}.jsonl"
        generation_manifest_path = root / f"{artifact_id}.manifest.json"
        model_sha256, tokenizer_sha256 = artifact_hashes[artifact_id]
        rows = []
        for sample_index, prompt_row in enumerate(prompts):
            sample_id = prompt_row["sample_id"]
            invalid = (artifact_id, sample_id) == invalid_key
            answer = "" if invalid else "Inflation declined while employment increased."
            rows.append(
                {
                    **prompt_row,
                    "artifact_id": artifact_id,
                    "verified_parent_artifact_id": parent,
                    "generation_model_sha256": model_sha256,
                    "generation_tokenizer_sha256": tokenizer_sha256,
                    "test_set_sha256": _sha_file(prompts_path),
                    "prompt_template_sha256": _sha_file(config_path),
                    "decoding_config_sha256": _sha_file(config_path),
                    "checkpoint_manifest_sha256": _sha_file(checkpoint_manifest_path),
                    "evaluation_config_sha256": _sha_file(config_path),
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 64,
                    "max_model_len": 512,
                    "generation_seed": derive_row_seed(20260729, sample_id),
                    "generation_seed_policy": "sample-id-sha256-v1",
                    "generated": answer,
                    "final_answer": answer,
                    "final_answer_sha256": _sha_text(answer),
                    "valid_generation": not invalid,
                    "invalid_reasons": (
                        [] if not invalid else ["empty_or_missing_generation"]
                    ),
                }
            )
        _write_jsonl(generation_path, rows)
        valid_count = sum(row["valid_generation"] for row in rows)
        generation_manifest = seal_manifest(
            {
                "schema_version": "checkpoint-artifact-generation-manifest-v1",
                "artifact_id": artifact_id,
                "sample_count": 33,
                "valid_count": valid_count,
                "invalid_count": 33 - valid_count,
                "checkpoint_manifest": {
                    "path": str(checkpoint_manifest_path),
                    "sha256": _sha_file(checkpoint_manifest_path),
                },
                "evaluation_config": {
                    "path": str(config_path),
                    "sha256": _sha_file(config_path),
                },
                "prompts": {
                    "path": str(prompts_path),
                    "sha256": _sha_file(prompts_path),
                    "row_count": 33,
                },
                "model_artifact": {"sha256": model_sha256},
                "tokenizer_artifact": {"sha256": tokenizer_sha256},
                "output": {
                    "path": str(generation_path),
                    "sha256": _sha_file(generation_path),
                    "row_count": 33,
                },
            }
        )
        _write_json(generation_manifest_path, generation_manifest)
        generation_paths.append(generation_path)
        generation_manifest_paths.append(generation_manifest_path)

    bertscore_dir = root / "bertscore"
    mpnet_dir = root / "mpnet"
    bertscore_dir.mkdir()
    mpnet_dir.mkdir()
    (bertscore_dir / "frozen.bin").write_bytes(b"bertscore fixture")
    (mpnet_dir / "frozen.bin").write_bytes(b"mpnet fixture")
    bertscore_sha256 = _sha256_path(bertscore_dir)
    mpnet_sha256 = _sha256_path(mpnet_dir)
    semantic_manifest = seal_manifest(
        {
            "schema_version": "checkpoint-eval-semantic-model-manifest-v1",
            "created_for_evaluation_id": "fixture-evaluation",
            "network_at_scoring_time": False,
            "models": {
                "bertscore": {
                    "repo_id": "fixture/bertscore",
                    "resolved_revision": "a" * 40,
                    "local_path": str(bertscore_dir),
                    "directory_sha256": bertscore_sha256,
                    "num_layers": 17,
                },
                "embedding_cosine": {
                    "repo_id": "fixture/mpnet",
                    "resolved_revision": "b" * 40,
                    "local_path": str(mpnet_dir),
                    "directory_sha256": mpnet_sha256,
                    "independent_from_training_reward": True,
                },
            },
        }
    )
    _write_json(semantic_manifest_path, semantic_manifest)

    return {
        "config": config_path,
        "prompts": prompts_path,
        "references": references_path,
        "test_manifest": test_manifest_path,
        "checkpoint_manifest": checkpoint_manifest_path,
        "semantic_manifest": semantic_manifest_path,
        "lineage_evidence": lineage_evidence_path,
        "semantic_scorers": [
            FakeBERTScoreBackend(bertscore_dir, bertscore_sha256),
            FakeMPNetBackend(mpnet_dir, mpnet_sha256),
        ],
        "generation_paths": generation_paths,
        "generation_manifests": generation_manifest_paths,
        "invalid_key": invalid_key,
    }


class TestFormalCheckpointEvaluation(unittest.TestCase):
    def test_formal_mode_preserves_exact_reference_leakage_check(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _build_formal_fixture(root)
            prompts = [
                json.loads(line)
                for line in fixture["prompts"].read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            references = [
                json.loads(line)
                for line in fixture["references"].read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            prompts[0]["prompt"] = (
                f"Evidence only: {references[0]['reference']} End evidence."
            )
            prompts[0]["prompt_sha256"] = _sha_text(prompts[0]["prompt"])
            _write_jsonl(fixture["prompts"], prompts)

            manifest = json.loads(
                fixture["test_manifest"].read_text(encoding="utf-8")
            )
            manifest.pop("integrity")
            manifest["outputs"]["prompts"]["sha256"] = _sha_file(
                fixture["prompts"]
            )
            _write_json(fixture["test_manifest"], seal_manifest(manifest))

            with self.assertRaisesRegex(
                ValueError,
                "Reference leakage detected",
            ):
                run_checkpoint_generation_evaluation(
                    fixture["generation_paths"],
                    fixture["references"],
                    fixture["prompts"],
                    root / "out",
                    generation_manifest_paths=fixture["generation_manifests"],
                    evaluation_config_path=fixture["config"],
                    test_manifest_path=fixture["test_manifest"],
                    checkpoint_manifest_path=fixture["checkpoint_manifest"],
                    semantic_manifest_path=fixture["semantic_manifest"],
                    lineage_evidence_path=fixture["lineage_evidence"],
                    semantic_scorers=fixture["semantic_scorers"],
                )

    def test_formal_mode_rejects_normalized_20_token_reference_leakage(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _build_formal_fixture(root)
            prompts = [
                json.loads(line)
                for line in fixture["prompts"].read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            references = [
                json.loads(line)
                for line in fixture["references"].read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            reference_tokens = [f"ReferenceToken{index}" for index in range(25)]
            reference_text = " ".join(reference_tokens)
            leaked_text = ", ".join(token.upper() for token in reference_tokens[:20])
            prompts[0]["prompt"] = f"Evidence only: {leaked_text}."
            prompts[0]["prompt_sha256"] = _sha_text(prompts[0]["prompt"])
            references[0]["reference"] = reference_text
            references[0]["reference_sha256"] = _sha_text(reference_text)
            _write_jsonl(fixture["prompts"], prompts)
            _write_jsonl(fixture["references"], references)

            manifest = json.loads(
                fixture["test_manifest"].read_text(encoding="utf-8")
            )
            manifest.pop("integrity")
            manifest["outputs"]["prompts"]["sha256"] = _sha_file(
                fixture["prompts"]
            )
            manifest["outputs"]["references"]["sha256"] = _sha_file(
                fixture["references"]
            )
            _write_json(fixture["test_manifest"], seal_manifest(manifest))

            with self.assertRaisesRegex(
                ValueError,
                "Normalized contiguous 20-token reference leakage",
            ):
                run_checkpoint_generation_evaluation(
                    fixture["generation_paths"],
                    fixture["references"],
                    fixture["prompts"],
                    root / "out",
                    generation_manifest_paths=fixture["generation_manifests"],
                    evaluation_config_path=fixture["config"],
                    test_manifest_path=fixture["test_manifest"],
                    checkpoint_manifest_path=fixture["checkpoint_manifest"],
                    semantic_manifest_path=fixture["semantic_manifest"],
                    lineage_evidence_path=fixture["lineage_evidence"],
                    semantic_scorers=fixture["semantic_scorers"],
                )

    def test_normalized_19_token_overlap_does_not_trigger_ngram_guard(self):
        reference_tokens = [f"ReferenceToken{index}" for index in range(25)]
        prompt = " / ".join(
            token.upper()
            for token in reference_tokens[
                : PROMPT_REFERENCE_OVERLAP_TOKEN_COUNT - 1
            ]
        )
        reference = " ".join(reference_tokens)

        self.assertIsNone(
            find_prompt_reference_token_overlap(prompt, reference)
        )

    def test_manifest_bound_33_row_matrix_and_prospective_subset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _build_formal_fixture(root)
            bertscore, mpnet = fixture["semantic_scorers"]

            result = run_checkpoint_generation_evaluation(
                fixture["generation_paths"],
                fixture["references"],
                fixture["prompts"],
                root / "out",
                generation_manifest_paths=fixture["generation_manifests"],
                evaluation_config_path=fixture["config"],
                test_manifest_path=fixture["test_manifest"],
                checkpoint_manifest_path=fixture["checkpoint_manifest"],
                semantic_manifest_path=fixture["semantic_manifest"],
                lineage_evidence_path=fixture["lineage_evidence"],
                semantic_scorers=fixture["semantic_scorers"],
            )

            self.assertEqual(
                len(result["row_scores"]),
                4 * FORMAL_TEST_SAMPLE_COUNT,
            )
            self.assertEqual(bertscore.candidate_count, 4 * 33 - 1)
            self.assertEqual(mpnet.candidate_count, 4 * 33 - 1)
            invalid = [
                row
                for row in result["row_scores"]
                if (row["artifact_id"], row["sample_id"]) == fixture["invalid_key"]
            ][0]
            self.assertEqual(invalid["status"], "invalid_output")
            self.assertEqual(invalid["valid_output"], 0.0)
            self.assertEqual(invalid["rouge_l_f1"], 0.0)
            self.assertEqual(invalid["bertscore_f1"], 0.0)
            self.assertEqual(invalid["mpnet_cosine"], 0.0)
            self.assertEqual(invalid["rule_covered_unsupported_rate"], 1.0)

            rouge_summaries = [
                row for row in result["summary"] if row["metric"] == "rouge_l_f1"
            ]
            self.assertEqual(
                {row["evaluation_subset"] for row in rouge_summaries},
                {"all_11_meetings", "prospective_only_9_meetings"},
            )
            prospective_rows = [
                row
                for row in rouge_summaries
                if row["evaluation_subset"] == "prospective_only_9_meetings"
            ]
            self.assertTrue(
                all(
                    row["n_rows_total"] == FORMAL_PROSPECTIVE_SAMPLE_COUNT
                    and row["n_meetings_total"] == 9
                    for row in prospective_rows
                )
            )
            self.assertTrue(
                all(
                    row["n_common_samples_total"] in {33, 27}
                    for row in result["contrasts"]
                )
            )
            self.assertEqual(
                {row["holm_family"] for row in result["contrasts"]},
                {
                    "all_11_meetings::all_prespecified_contrast_metric_tests",
                    (
                        "prospective_only_9_meetings::"
                        "all_prespecified_contrast_metric_tests"
                    ),
                },
            )
            formal_audit = result["audit"]["formal_validation"]
            self.assertTrue(formal_audit["complete"])
            self.assertTrue(formal_audit["invalid_rows_preserved"])
            self.assertTrue(formal_audit["semantic_models"]["complete"])
            self.assertEqual(
                result["audit"]["run_spec"]["semantic_manifest"]["sha256"],
                _sha_file(fixture["semantic_manifest"]),
            )
            self.assertTrue(formal_audit["lineage_evidence"]["complete"])
            self.assertFalse(
                formal_audit["lineage_evidence"]["tensor_comparison_recomputed"]
            )
            self.assertEqual(
                result["audit"]["run_spec"]["lineage_evidence"]["sha256"],
                _sha_file(fixture["lineage_evidence"]),
            )
            self.assertTrue(
                all(
                    item["inference_environment_audit"]["status"]
                    == "not_recorded_legacy_compatible"
                    for item in formal_audit["generation_manifests"]
                )
            )

    def test_formal_length_tolerant_policy_scores_nonempty_token_limit_row(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _build_formal_fixture(root)
            artifact_id, sample_id = fixture["invalid_key"]
            generation_path = next(
                path
                for path in fixture["generation_paths"]
                if path.stem == artifact_id
            )
            manifest_path = next(
                path
                for path in fixture["generation_manifests"]
                if path.name == f"{artifact_id}.manifest.json"
            )
            generation_rows = [
                json.loads(line)
                for line in generation_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            for row in generation_rows:
                if row["sample_id"] != sample_id:
                    continue
                row["generated"] = (
                    "<think>unfinished"
                    "<answer>Inflation declined while employment increased."
                )
                row["generation_finish_reason"] = "length"
                row["invalid_reasons"] = ["non_normal_finish:length"]
                row["input_was_truncated"] = False
                break
            _write_jsonl(generation_path, generation_rows)
            generation_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            generation_manifest.pop("integrity")
            generation_manifest["output"]["sha256"] = _sha_file(generation_path)
            _write_json(manifest_path, seal_manifest(generation_manifest))

            result = run_checkpoint_generation_evaluation(
                fixture["generation_paths"],
                fixture["references"],
                fixture["prompts"],
                root / "out",
                generation_manifest_paths=fixture["generation_manifests"],
                evaluation_config_path=fixture["config"],
                test_manifest_path=fixture["test_manifest"],
                checkpoint_manifest_path=fixture["checkpoint_manifest"],
                semantic_manifest_path=fixture["semantic_manifest"],
                lineage_evidence_path=fixture["lineage_evidence"],
                semantic_scorers=fixture["semantic_scorers"],
                scoring_policy=LENGTH_TOLERANT_SCORING_POLICY,
            )

            self.assertEqual(
                sum(row["valid_output"] for row in result["row_scores"]),
                4 * FORMAL_TEST_SAMPLE_COUNT,
            )
            recovered = [
                row
                for row in result["row_scores"]
                if (row["artifact_id"], row["sample_id"]) == fixture["invalid_key"]
            ][0]
            self.assertEqual(recovered["status"], "scored")
            self.assertEqual(recovered["candidate_extraction_mode"], "answer_tag")
            self.assertTrue(recovered["source_generation_hit_token_limit"])
            self.assertEqual(
                result["audit"]["scoring_policy"]["policy_id"],
                LENGTH_TOLERANT_SCORING_POLICY,
            )
            self.assertTrue(
                result["audit"]["scoring_policy_row_audit"][
                    "all_declared_rows_valid"
                ]
            )

    def test_tampered_generation_manifest_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _build_formal_fixture(root)
            manifest_path = fixture["generation_manifests"][0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["valid_count"] -= 1
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(ValueError, "integrity validation failed"):
                run_checkpoint_generation_evaluation(
                    fixture["generation_paths"],
                    fixture["references"],
                    fixture["prompts"],
                    root / "out",
                    generation_manifest_paths=fixture["generation_manifests"],
                    evaluation_config_path=fixture["config"],
                    test_manifest_path=fixture["test_manifest"],
                    checkpoint_manifest_path=fixture["checkpoint_manifest"],
                    semantic_manifest_path=fixture["semantic_manifest"],
                    lineage_evidence_path=fixture["lineage_evidence"],
                    semantic_scorers=fixture["semantic_scorers"],
                )

    def test_formal_mode_requires_both_manifest_bound_semantic_backends(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _build_formal_fixture(root)
            with self.assertRaisesRegex(
                ValueError,
                "exactly one frozen BERTScore scorer",
            ):
                run_checkpoint_generation_evaluation(
                    fixture["generation_paths"],
                    fixture["references"],
                    fixture["prompts"],
                    root / "out",
                    generation_manifest_paths=fixture["generation_manifests"],
                    evaluation_config_path=fixture["config"],
                    test_manifest_path=fixture["test_manifest"],
                    checkpoint_manifest_path=fixture["checkpoint_manifest"],
                    semantic_manifest_path=fixture["semantic_manifest"],
                    lineage_evidence_path=fixture["lineage_evidence"],
                    semantic_scorers=fixture["semantic_scorers"][:1],
                )

    def test_semantic_manifest_num_layers_must_match_backend(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _build_formal_fixture(root)
            semantic_path = fixture["semantic_manifest"]
            semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
            semantic.pop("integrity")
            semantic["models"]["bertscore"]["num_layers"] = 16
            _write_json(semantic_path, seal_manifest(semantic))
            with self.assertRaisesRegex(ValueError, "num_layers differs"):
                run_checkpoint_generation_evaluation(
                    fixture["generation_paths"],
                    fixture["references"],
                    fixture["prompts"],
                    root / "out",
                    generation_manifest_paths=fixture["generation_manifests"],
                    evaluation_config_path=fixture["config"],
                    test_manifest_path=fixture["test_manifest"],
                    checkpoint_manifest_path=fixture["checkpoint_manifest"],
                    semantic_manifest_path=fixture["semantic_manifest"],
                    lineage_evidence_path=fixture["lineage_evidence"],
                    semantic_scorers=fixture["semantic_scorers"],
                )

    def test_generation_seed_is_recomputed_from_base_seed_and_sample_id(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _build_formal_fixture(root)
            generation_path = fixture["generation_paths"][0]
            rows = [
                json.loads(line)
                for line in generation_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["generation_seed"] += 1
            _write_jsonl(generation_path, rows)
            manifest_path = fixture["generation_manifests"][0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("integrity")
            manifest["output"]["sha256"] = _sha_file(generation_path)
            _write_json(manifest_path, seal_manifest(manifest))

            with self.assertRaisesRegex(ValueError, "generation_seed"):
                run_checkpoint_generation_evaluation(
                    fixture["generation_paths"],
                    fixture["references"],
                    fixture["prompts"],
                    root / "out",
                    generation_manifest_paths=fixture["generation_manifests"],
                    evaluation_config_path=fixture["config"],
                    test_manifest_path=fixture["test_manifest"],
                    checkpoint_manifest_path=fixture["checkpoint_manifest"],
                    semantic_manifest_path=fixture["semantic_manifest"],
                    lineage_evidence_path=fixture["lineage_evidence"],
                    semantic_scorers=fixture["semantic_scorers"],
                )

    def test_lineage_evidence_checkpoint_hash_binding_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _build_formal_fixture(root)
            evidence_path = fixture["lineage_evidence"]
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence.pop("integrity")
            evidence["bindings"]["checkpoint_manifest"]["sha256"] = "0" * 64
            _write_json(evidence_path, seal_manifest(evidence))

            with self.assertRaisesRegex(ValueError, "hash binding changed"):
                run_checkpoint_generation_evaluation(
                    fixture["generation_paths"],
                    fixture["references"],
                    fixture["prompts"],
                    root / "out",
                    generation_manifest_paths=fixture["generation_manifests"],
                    evaluation_config_path=fixture["config"],
                    test_manifest_path=fixture["test_manifest"],
                    checkpoint_manifest_path=fixture["checkpoint_manifest"],
                    semantic_manifest_path=fixture["semantic_manifest"],
                    lineage_evidence_path=fixture["lineage_evidence"],
                    semantic_scorers=fixture["semantic_scorers"],
                )

    def test_lineage_evidence_tensor_summary_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _build_formal_fixture(root)
            evidence_path = fixture["lineage_evidence"]
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence.pop("integrity")
            evidence["tensor_verification"]["mismatch_count"] = 1
            _write_json(evidence_path, seal_manifest(evidence))

            with self.assertRaisesRegex(ValueError, "tensor summary"):
                run_checkpoint_generation_evaluation(
                    fixture["generation_paths"],
                    fixture["references"],
                    fixture["prompts"],
                    root / "out",
                    generation_manifest_paths=fixture["generation_manifests"],
                    evaluation_config_path=fixture["config"],
                    test_manifest_path=fixture["test_manifest"],
                    checkpoint_manifest_path=fixture["checkpoint_manifest"],
                    semantic_manifest_path=fixture["semantic_manifest"],
                    lineage_evidence_path=fixture["lineage_evidence"],
                    semantic_scorers=fixture["semantic_scorers"],
                )


if __name__ == "__main__":
    unittest.main()
