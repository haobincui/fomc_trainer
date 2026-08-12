import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jobs.eval.eval_checkpoint_generation import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    LENGTH_TOLERANT_SCORING_POLICY,
    align_artifacts,
    check_format,
    cluster_bootstrap_interval,
    contrast_rows,
    evaluate_factual_rules,
    holm_adjust,
    repetition_rate,
    rouge_l_f1,
    run_checkpoint_generation_evaluation,
    score_rows,
    summarise_rows,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class InjectedSemanticScorer:
    scorer_id = "fake-independent"
    model_id = "fixture-encoder-v1"
    model_sha256 = "f" * 64

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    def score(self, candidates, references):
        self.calls.append((list(candidates), list(references)))
        return {
            "fake_semantic": [
                1.0 if candidate == reference else 0.25
                for candidate, reference in zip(
                    candidates,
                    references,
                    strict=True,
                )
            ]
        }


class TestOuterAlignment(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_rows = [
            {"sample_id": "s1", "meeting_id": "m1", "reference": "one"},
            {"sample_id": "s2", "meeting_id": "m2", "reference": "two"},
        ]
        self.evidence_rows = [
            {"sample_id": "s1", "meeting_id": "m1", "evidence_text": "one"},
            {"sample_id": "s2", "meeting_id": "m2", "evidence_text": "two"},
        ]

    def test_requires_exactly_four_artifacts_by_default(self):
        generations = [
            {
                "artifact_id": artifact,
                "sample_id": sample,
                "meeting_id": f"m{sample[-1]}",
                "generated": sample,
            }
            for artifact in ("a", "b", "c")
            for sample in ("s1", "s2")
        ]

        with self.assertRaisesRegex(ValueError, "exactly 4"):
            align_artifacts(
                generations,
                self.reference_rows,
                self.evidence_rows,
            )

    def test_outer_join_refuses_a_missing_generation(self):
        generations = [
            {
                "artifact_id": artifact,
                "sample_id": sample,
                "meeting_id": f"m{sample[-1]}",
                "generated": sample,
            }
            for artifact in ("a", "b", "c", "d")
            for sample in ("s1", "s2")
            if not (artifact == "d" and sample == "s2")
        ]

        with self.assertRaisesRegex(
            ValueError,
            "refusing a silent inner join",
        ) as raised:
            align_artifacts(
                generations,
                self.reference_rows,
                self.evidence_rows,
            )

        self.assertIn('"d": ["s2"]', str(raised.exception))

    def test_record_mode_retains_missing_cell_and_audits_it(self):
        generations = [
            {
                "artifact_id": artifact,
                "sample_id": sample,
                "meeting_id": f"m{sample[-1]}",
                "generated": sample,
            }
            for artifact in ("a", "b", "c", "d")
            for sample in ("s1", "s2")
            if not (artifact == "d" and sample == "s2")
        ]

        aligned, audit = align_artifacts(
            generations,
            self.reference_rows,
            self.evidence_rows,
            missing_policy="record",
        )

        self.assertEqual(len(aligned), 8)
        missing = [
            row
            for row in aligned
            if row["artifact_id"] == "d" and row["sample_id"] == "s2"
        ][0]
        self.assertEqual(missing["alignment_status"], "missing_generation")
        self.assertEqual(
            audit["missing_generation_sample_ids_by_artifact"]["d"], ["s2"]
        )
        self.assertFalse(audit["complete"])

    def test_duplicate_generation_key_is_rejected(self):
        generations = [
            {
                "artifact_id": "a",
                "sample_id": "s1",
                "meeting_id": "m1",
                "generated": "first",
            },
            {
                "artifact_id": "a",
                "sample_id": "s1",
                "meeting_id": "m1",
                "generated": "second",
            },
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate generation"):
            align_artifacts(
                generations,
                self.reference_rows[:1],
                self.evidence_rows[:1],
                expected_artifact_count=1,
            )


class TestSurfaceAndRuleMetrics(unittest.TestCase):
    def test_rouge_l_and_repetition_are_local_and_deterministic(self):
        self.assertAlmostEqual(rouge_l_f1("a b c", "a x c"), 2 / 3)
        self.assertEqual(repetition_rate("a b"), 0.0)
        self.assertAlmostEqual(
            repetition_rate("a b c a b c"),
            1 / 4,
        )

    def test_formal_body_only_format_defaults_reject_tags_evidence_and_headings(self):
        score, violations = check_format(
            (
                "<answer>Staff Review of the Financial Situation</answer>\n"
                "<<<BEGIN-D1-EVIDENCE>>>"
            ),
            {"valid_generation": True},
            {"section_name": "Staff Review of the Financial Situation"},
            {},
        )

        self.assertEqual(score, 0.0)
        self.assertIn(
            "formal_body_only_violation:reasoning_or_answer_tag",
            violations,
        )
        self.assertIn(
            "formal_body_only_violation:d1_evidence_delimiter",
            violations,
        )
        self.assertIn(
            "formal_body_only_violation:financial_situation_section_heading",
            violations,
        )

    def test_supported_numeric_unit_time_direction_and_stance(self):
        evidence = {
            "facts": [
                {
                    "fact_id": "inflation-2024",
                    "topic": "inflation",
                    "value": 3,
                    "unit": "percent",
                    "time_range": "2024",
                    "direction": "down",
                }
            ],
            "policy_stance": "hawkish",
        }

        metrics, checks = evaluate_factual_rules(
            (
                "Inflation declined to 3 percent in 2024. "
                "Officials favored a restrictive stance."
            ),
            evidence,
        )

        self.assertEqual(metrics["numeric_value_accuracy"], 1.0)
        self.assertEqual(metrics["unit_accuracy"], 1.0)
        self.assertEqual(metrics["time_accuracy"], 1.0)
        self.assertEqual(metrics["novel_number_rate"], 0.0)
        self.assertEqual(metrics["direction_consistency"], 1.0)
        self.assertEqual(metrics["direction_coverage"], 1.0)
        self.assertEqual(metrics["policy_stance_consistency"], 1.0)
        self.assertEqual(metrics["rule_covered_unsupported_rate"], 0.0)
        self.assertTrue(checks)
        self.assertTrue(all("counts_toward_unsupported" in row for row in checks))

    def test_value_and_unit_accuracy_are_separate(self):
        evidence = {
            "facts": [
                {
                    "topic": "inflation",
                    "value": 3,
                    "unit": "percent",
                    "time_range": "2024",
                    "direction": "down",
                }
            ],
            "policy_stance": "hawkish",
        }

        metrics, checks = evaluate_factual_rules(
            ("Inflation rose to 3 basis points in 2023. Officials favored rate cuts."),
            evidence,
        )

        self.assertEqual(metrics["numeric_value_accuracy"], 1.0)
        self.assertEqual(metrics["unit_accuracy"], 0.0)
        self.assertEqual(metrics["time_accuracy"], 0.0)
        self.assertGreater(metrics["novel_number_rate"], 0.0)
        self.assertEqual(metrics["direction_consistency"], 0.0)
        self.assertEqual(metrics["policy_stance_consistency"], 0.0)
        self.assertGreater(metrics["rule_covered_unsupported_rate"], 0.0)
        reasons = {row["reason"] for row in checks if not row["supported"]}
        self.assertIn("unit_missing_or_conflicts_with_evidence", reasons)
        self.assertIn("direction_conflicts_with_evidence", reasons)
        self.assertIn("stance_conflicts_with_evidence", reasons)

    def test_percent_percentage_point_and_basis_point_are_distinct_units(self):
        percent_metrics, _ = evaluate_factual_rules(
            "Inflation was 3 percentage points.",
            {"facts": [{"topic": "inflation", "value": 3, "unit": "percent"}]},
        )
        percentage_point_metrics, _ = evaluate_factual_rules(
            "The rate changed 0.25 percent.",
            {
                "facts": [
                    {
                        "topic": "policy",
                        "value": 0.25,
                        "unit": "percentage points",
                    }
                ]
            },
        )
        basis_point_metrics, _ = evaluate_factual_rules(
            "The rate changed 25 basis points.",
            {
                "facts": [
                    {
                        "topic": "policy",
                        "value": 25,
                        "unit": "percentage points",
                    }
                ]
            },
        )
        no_conversion_metrics, _ = evaluate_factual_rules(
            "The rate changed 25 basis points.",
            {
                "facts": [
                    {
                        "topic": "policy",
                        "value": 0.25,
                        "unit": "percentage points",
                    }
                ]
            },
        )

        self.assertEqual(percent_metrics["numeric_value_accuracy"], 1.0)
        self.assertEqual(percent_metrics["unit_accuracy"], 0.0)
        self.assertEqual(percentage_point_metrics["numeric_value_accuracy"], 1.0)
        self.assertEqual(percentage_point_metrics["unit_accuracy"], 0.0)
        self.assertEqual(basis_point_metrics["numeric_value_accuracy"], 1.0)
        self.assertEqual(basis_point_metrics["unit_accuracy"], 0.0)
        self.assertEqual(no_conversion_metrics["numeric_value_accuracy"], 0.0)

    def test_numeric_match_uses_declared_display_precision_with_audit(self):
        metrics, checks = evaluate_factual_rules(
            "Payrolls were 23,861 thousand and the ratio was 0.0244 percent.",
            {
                "facts": [
                    {
                        "fact_id": "payroll-level",
                        "value": "23860.7342",
                        "unit": "thousand",
                    },
                    {
                        "fact_id": "ratio-value",
                        "value": "0.024377",
                        "unit": "percent",
                    },
                ]
            },
        )

        self.assertEqual(metrics["numeric_value_accuracy"], 1.0)
        self.assertEqual(metrics["unit_accuracy"], 1.0)
        numeric_checks = [
            check for check in checks if check["rule_kind"] == "numeric"
        ]
        self.assertEqual(
            [check["matched_fact_ids"] for check in numeric_checks],
            [["payroll-level"], ["ratio-value"]],
        )
        self.assertEqual(
            [check["display_decimal_places"] for check in numeric_checks],
            [0, 4],
        )
        self.assertTrue(
            all(
                check["numeric_match_policy"]
                == "decimal-display-precision-round-half-up-v1"
                for check in numeric_checks
            )
        )

    def test_numeric_display_precision_has_deterministic_half_up_boundary(self):
        matched, _ = evaluate_factual_rules(
            "The estimate was 1.23 percent.",
            {"facts": [{"value": "1.2349", "unit": "percent"}]},
        )
        rejected_positive_tie, _ = evaluate_factual_rules(
            "The estimate was 1.23 percent.",
            {"facts": [{"value": "1.235", "unit": "percent"}]},
        )
        rejected_negative_tie, _ = evaluate_factual_rules(
            "The estimate was -1.23 percent.",
            {"facts": [{"value": "-1.235", "unit": "percent"}]},
        )

        self.assertEqual(matched["numeric_value_accuracy"], 1.0)
        self.assertEqual(rejected_positive_tie["numeric_value_accuracy"], 0.0)
        self.assertEqual(rejected_negative_tie["numeric_value_accuracy"], 0.0)

    def test_directions_are_attached_to_nearest_topic_in_same_sentence(self):
        metrics, checks = evaluate_factual_rules(
            "Inflation declined while growth rose.",
            {
                "directions": {
                    "inflation": "down",
                    "growth": "up",
                }
            },
        )

        direction_checks = [row for row in checks if row["rule_kind"] == "direction"]
        self.assertEqual(len(direction_checks), 2)
        self.assertEqual(metrics["direction_consistency"], 1.0)

    def test_explicit_custom_rules_define_scope_of_unsupported_rate(self):
        metrics, checks = evaluate_factual_rules(
            "Demand came from extraterrestrial purchases.",
            {
                "claim_rules": [
                    {
                        "claim_id": "alien-demand",
                        "pattern": "extraterrestrial purchases",
                        "supported": False,
                    }
                ]
            },
        )

        self.assertEqual(metrics["rule_covered_claim_count"], 1)
        self.assertEqual(metrics["rule_covered_unsupported_rate"], 1.0)
        self.assertEqual(checks[0]["rule_kind"], "custom")

    def test_directions_and_stance_are_derived_from_formal_evidence_facts(self):
        facts = [
            {
                "indicator": "Consumer-Price-Index-(CPI)",
                "series_id": "CPI",
                "kind": "derived_absolute_change",
                "value": "-0.1",
                "unit": "Percent",
            },
            {
                "indicator": "GDP-Growth",
                "series_id": "GDP",
                "kind": "derived_absolute_change",
                "value": "0.2",
                "unit": "Percent",
            },
            {
                "indicator": "Labour-Market",
                "series_id": "PAYEMS",
                "kind": "derived_absolute_change",
                "value": "1",
                "unit": "Thousands of Persons",
            },
            {
                "indicator": "Federal-Funds-Rate",
                "series_id": "DFF",
                "kind": "derived_absolute_change",
                "value": "-0.25",
                "unit": "percentage points",
            },
        ]

        metrics, _ = evaluate_factual_rules(
            (
                "Inflation declined, growth rose, and employment increased. "
                "The policy stance shifted toward easing."
            ),
            {"evidence_facts": facts},
        )

        self.assertEqual(metrics["expected_direction_topic_count"], 3)
        self.assertEqual(metrics["direction_consistency"], 1.0)
        self.assertEqual(metrics["direction_coverage"], 1.0)
        self.assertEqual(metrics["expected_policy_stance_available"], 1)
        self.assertEqual(metrics["policy_stance_consistency"], 1.0)
        self.assertEqual(metrics["policy_stance_coverage"], 1.0)

    def test_claim_without_rule_coverable_evidence_is_not_marked_supported(self):
        metrics, checks = evaluate_factual_rules(
            "Inflation rose and policy tightening followed.",
            {"evidence_facts": []},
        )

        covered = [check for check in checks if check["counts_toward_unsupported"]]
        self.assertTrue(covered)
        self.assertTrue(all(not check["supported"] for check in covered))
        self.assertEqual(metrics["rule_covered_unsupported_rate"], 1.0)
        self.assertEqual(metrics["expected_policy_stance_available"], 0)
        self.assertIsNone(metrics["policy_stance_coverage"])
        self.assertIsNone(metrics["policy_stance_consistency"])

    def test_conflicting_short_run_direction_evidence_is_excluded_from_denominator(
        self,
    ):
        metrics, checks = evaluate_factual_rules(
            "Inflation rose.",
            {
                "evidence_facts": [
                    {
                        "indicator": "Consumer-Price-Index-(CPI)",
                        "series_id": "headline",
                        "kind": "derived_absolute_change",
                        "value": "0.1",
                        "unit": "Percent",
                    },
                    {
                        "indicator": "Personal-Consumption-Expenditures-(PCE)",
                        "series_id": "core",
                        "kind": "derived_absolute_change",
                        "value": "-0.1",
                        "unit": "Percent",
                    },
                ]
            },
        )

        direction_checks = [
            check for check in checks if check["rule_kind"] == "direction"
        ]
        self.assertEqual(metrics["expected_direction_topic_count"], 0)
        self.assertEqual(metrics["direction_coverable_claim_count"], 0)
        self.assertIsNone(metrics["direction_consistency"])
        self.assertIsNone(metrics["direction_coverage"])
        self.assertEqual(
            direction_checks[0]["reason"], "direction_evidence_unavailable"
        )
        self.assertFalse(direction_checks[0]["supported"])


class TestMeetingAggregationAndInference(unittest.TestCase):
    def test_summary_equal_weights_meetings_not_rows(self):
        rows = [
            {
                "artifact_id": "a",
                "sample_id": "s1",
                "meeting_id": "m1",
                "status": "scored",
                "rouge_l_f1": 1.0,
            },
            {
                "artifact_id": "a",
                "sample_id": "s2",
                "meeting_id": "m1",
                "status": "scored",
                "rouge_l_f1": 1.0,
            },
            {
                "artifact_id": "a",
                "sample_id": "s3",
                "meeting_id": "m2",
                "status": "invalid_output",
                "rouge_l_f1": 0.0,
            },
        ]

        summaries = summarise_rows(
            rows,
            bootstrap_samples=100,
            bootstrap_seed=9,
        )
        rouge = [row for row in summaries if row["metric"] == "rouge_l_f1"][0]

        self.assertEqual(rouge["mean"], 0.5)
        self.assertEqual(rouge["n_rows_eligible"], 3)
        self.assertEqual(rouge["n_meetings"], 2)
        self.assertEqual(rouge["n_invalid_output"], 1)

    def test_cluster_bootstrap_is_deterministic(self):
        values = {"m1": 0.1, "m2": 0.4, "m3": 0.8}

        first = cluster_bootstrap_interval(values, samples=500, seed=20260729)
        second = cluster_bootstrap_interval(values, samples=500, seed=20260729)

        self.assertEqual(first, second)
        self.assertLess(first[0], first[1])
        self.assertEqual(DEFAULT_BOOTSTRAP_SAMPLES, 10_000)
        self.assertEqual(DEFAULT_BOOTSTRAP_SEED, 20_260_729)

    def test_holm_adjustment_is_monotone_in_sorted_p_values(self):
        adjusted = holm_adjust([0.01, 0.04, 0.02, None])

        self.assertEqual(adjusted, [0.03, 0.04, 0.04, None])

    def test_only_lineage_legal_contrast_is_accepted(self):
        rows = []
        for artifact, values in {
            "base": (0.1, 0.2),
            "sft": (0.3, 0.4),
            "minutes": (0.5, 0.6),
        }.items():
            for index, value in enumerate(values, start=1):
                rows.append(
                    {
                        "artifact_id": artifact,
                        "sample_id": f"s{index}",
                        "meeting_id": f"m{index}",
                        "rouge_l_f1": value,
                        "generation_seed": 7,
                    }
                )
        provenance = {
            artifact: {
                "test_set_sha256": "a" * 64,
                "prompt_template_sha256": "b" * 64,
                "decoding_config_sha256": "c" * 64,
                "parent_artifact_id": parent,
            }
            for artifact, parent in {
                "base": None,
                "sft": "base",
                "minutes": "sft",
            }.items()
        }
        lineage = {"base": None, "sft": "base", "minutes": "sft"}

        valid, plan = contrast_rows(
            rows,
            provenance=provenance,
            lineage=lineage,
            contrasts=[("base", "sft")],
            metrics=["rouge_l_f1"],
            bootstrap_samples=100,
            bootstrap_seed=11,
        )

        self.assertEqual(len(plan), 1)
        self.assertEqual(valid[0]["paired_mean_difference"], 0.2)
        self.assertEqual(valid[0]["n_meetings"], 2)
        self.assertIsNotNone(valid[0]["p_value_holm"])

        with self.assertRaisesRegex(ValueError, "Illegal parent_child contrast"):
            contrast_rows(
                rows,
                provenance=provenance,
                lineage=lineage,
                contrasts=[("base", "minutes")],
                metrics=["rouge_l_f1"],
                bootstrap_samples=10,
            )

    def test_contrast_rejects_per_sample_decoding_mismatch(self):
        rows = [
            {
                "artifact_id": "base",
                "sample_id": "s1",
                "meeting_id": "m1",
                "rouge_l_f1": 0.1,
                "generation_seed": 1,
            },
            {
                "artifact_id": "sft",
                "sample_id": "s1",
                "meeting_id": "m1",
                "rouge_l_f1": 0.2,
                "generation_seed": 2,
            },
        ]
        provenance = {
            artifact: {
                "test_set_sha256": "a" * 64,
                "prompt_template_sha256": "b" * 64,
                "decoding_config_sha256": "c" * 64,
                "parent_artifact_id": parent,
            }
            for artifact, parent in {"base": None, "sft": "base"}.items()
        }

        with self.assertRaisesRegex(ValueError, "generation_seed"):
            contrast_rows(
                rows,
                provenance=provenance,
                lineage={"base": None, "sft": "base"},
                contrasts=[("base", "sft")],
                metrics=["rouge_l_f1"],
                bootstrap_samples=10,
            )


class TestInjectedSemanticBackend(unittest.TestCase):
    def test_injected_scorer_is_batched_without_loading_a_model(self):
        aligned = [
            {
                "artifact_id": "a",
                "sample_id": "s1",
                "meeting_id": "m1",
                "generation_row": {
                    "generated": "same",
                    "generation_validation_status": "passed",
                },
                "reference_row": {"reference": "same"},
                "evidence_row": {"evidence_text": "same"},
                "alignment_status": "complete",
                "missing_inputs": [],
            },
            {
                "artifact_id": "a",
                "sample_id": "s2",
                "meeting_id": "m2",
                "generation_row": {
                    "generated": "different",
                    "generation_validation_status": "passed",
                },
                "reference_row": {"reference": "target"},
                "evidence_row": {"evidence_text": "target"},
                "alignment_status": "complete",
                "missing_inputs": [],
            },
        ]
        scorer = InjectedSemanticScorer()

        rows, _, metadata = score_rows(aligned, semantic_scorers=[scorer])

        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual([row["fake_semantic"] for row in rows], [1.0, 0.25])
        self.assertEqual(metadata[0]["model_sha256"], "f" * 64)
        self.assertTrue(metadata[0]["injected"])

    def test_nonfatal_format_violation_is_still_fully_scored(self):
        generated = (
            "<answer>Staff Review of the Economic Situation. "
            "Inflation was 3 percent.</answer>"
        )
        aligned = [
            {
                "artifact_id": "a",
                "sample_id": "s1",
                "meeting_id": "m1",
                "generation_row": {
                    "generated": generated,
                    "generation_validation_status": "passed",
                    "valid_generation": True,
                },
                "reference_row": {"reference": generated},
                "evidence_row": {
                    "facts": [
                        {
                            "fact_id": "inflation",
                            "value": 3,
                            "unit": "percent",
                        }
                    ]
                },
                "alignment_status": "complete",
                "missing_inputs": [],
            }
        ]
        scorer = InjectedSemanticScorer()

        rows, _, _ = score_rows(aligned, semantic_scorers=[scorer])
        row = rows[0]

        self.assertEqual(row["status"], "scored")
        self.assertEqual(row["valid_output"], 1.0)
        self.assertFalse(row["fatal_generation_invalid"])
        self.assertEqual(row["invalid_reasons"], [])
        self.assertEqual(row["format_compliance"], 0.0)
        self.assertTrue(row["format_violations"])
        self.assertEqual(row["rouge_l_f1"], 1.0)
        self.assertEqual(row["numeric_value_accuracy"], 1.0)
        self.assertEqual(row["fake_semantic"], 1.0)
        self.assertEqual(len(scorer.calls), 1)
        format_summary = [
            item
            for item in summarise_rows(rows, bootstrap_samples=10, bootstrap_seed=7)
            if item["metric"] == "format_compliance"
        ][0]
        self.assertEqual(format_summary["n_valid_output"], 1)
        self.assertEqual(format_summary["n_invalid_output"], 0)

    def test_fatal_token_limit_output_gets_worst_case_and_skips_semantic(self):
        aligned = [
            {
                "artifact_id": "a",
                "sample_id": "s1",
                "meeting_id": "m1",
                "generation_row": {
                    "generated": "Inflation was 3 percent.",
                    "generation_finish_reason": "length",
                },
                "reference_row": {"reference": "Inflation was 3 percent."},
                "evidence_row": {
                    "facts": [{"value": 3, "unit": "percent"}],
                },
                "alignment_status": "complete",
                "missing_inputs": [],
            }
        ]
        scorer = InjectedSemanticScorer()

        rows, _, _ = score_rows(aligned, semantic_scorers=[scorer])
        row = rows[0]

        self.assertEqual(row["status"], "invalid_output")
        self.assertEqual(row["valid_output"], 0.0)
        self.assertTrue(row["fatal_generation_invalid"])
        self.assertIn("truncated:length", row["invalid_reasons"])
        self.assertEqual(row["rouge_l_f1"], 0.0)
        self.assertEqual(row["numeric_value_accuracy"], 0.0)
        self.assertNotIn("fake_semantic", row)
        self.assertEqual(scorer.calls, [])

    def test_length_tolerant_policy_scores_answer_after_opening_tag(self):
        aligned = [
            {
                "artifact_id": "a",
                "sample_id": "s1",
                "meeting_id": "m1",
                "generation_row": {
                    "generated": (
                        "<think>unfinished reasoning "
                        "<answer>Inflation was 3 percent.</answer>"
                    ),
                    "final_answer": "",
                    "generation_finish_reason": "length",
                    "valid_generation": False,
                    "invalid_reasons": ["non_normal_finish:length"],
                    "input_was_truncated": False,
                },
                "reference_row": {"reference": "Inflation was 3 percent."},
                "evidence_row": {
                    "facts": [{"value": 3, "unit": "percent"}],
                },
                "alignment_status": "complete",
                "missing_inputs": [],
            }
        ]
        scorer = InjectedSemanticScorer()

        rows, _, _ = score_rows(
            aligned,
            semantic_scorers=[scorer],
            scoring_policy=LENGTH_TOLERANT_SCORING_POLICY,
        )
        row = rows[0]

        self.assertEqual(row["status"], "scored")
        self.assertEqual(row["valid_output"], 1.0)
        self.assertEqual(row["generated_text"], "Inflation was 3 percent.")
        self.assertEqual(row["candidate_extraction_mode"], "answer_tag")
        self.assertTrue(row["has_leading_think_opening_tag"])
        self.assertTrue(row["has_answer_opening_tag"])
        self.assertTrue(row["has_trailing_answer_closing_tag"])
        self.assertTrue(row["source_generation_hit_token_limit"])
        self.assertEqual(
            row["ignored_upstream_invalid_reasons"],
            ["non_normal_finish:length"],
        )
        self.assertEqual(row["rouge_l_f1"], 1.0)
        self.assertEqual(row["numeric_value_accuracy"], 1.0)
        self.assertEqual(row["fake_semantic"], 1.0)

    def test_length_tolerant_policy_does_not_require_closing_think_or_answer_tag(
        self,
    ):
        generated = "<think>Inflation was 3 percent."
        aligned = [
            {
                "artifact_id": "a",
                "sample_id": "s1",
                "meeting_id": "m1",
                "generation_row": {
                    "generated": generated,
                    "generation_finish_reason": "length",
                    "valid_generation": False,
                    "invalid_reasons": ["non_normal_finish:length"],
                    "input_was_truncated": False,
                },
                "reference_row": {"reference": generated},
                "evidence_row": {"facts": [{"value": 3, "unit": "percent"}]},
                "alignment_status": "complete",
                "missing_inputs": [],
            }
        ]

        rows, _, _ = score_rows(
            aligned,
            scoring_policy=LENGTH_TOLERANT_SCORING_POLICY,
        )

        self.assertEqual(rows[0]["valid_output"], 1.0)
        self.assertEqual(rows[0]["candidate_extraction_mode"], "full_completion")
        self.assertEqual(rows[0]["generated_text"], generated)
        self.assertEqual(rows[0]["rouge_l_f1"], 1.0)

    def test_length_tolerant_policy_still_rejects_input_truncation(self):
        aligned = [
            {
                "artifact_id": "a",
                "sample_id": "s1",
                "meeting_id": "m1",
                "generation_row": {
                    "generated": "Inflation was 3 percent.",
                    "generation_finish_reason": "length",
                    "valid_generation": False,
                    "invalid_reasons": ["non_normal_finish:length"],
                    "input_was_truncated": True,
                },
                "reference_row": {"reference": "Inflation was 3 percent."},
                "evidence_row": {"facts": [{"value": 3, "unit": "percent"}]},
                "alignment_status": "complete",
                "missing_inputs": [],
            }
        ]

        rows, _, _ = score_rows(
            aligned,
            scoring_policy=LENGTH_TOLERANT_SCORING_POLICY,
        )

        self.assertEqual(rows[0]["valid_output"], 0.0)
        self.assertIn("input_was_truncated", rows[0]["invalid_reasons"])


class TestEndToEndEvaluator(unittest.TestCase):
    def test_writes_all_auditable_outputs_without_model_inference(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference_path = root / "reference.jsonl"
            evidence_path = root / "evidence.jsonl"
            generation_path = root / "generations.jsonl"
            lineage_path = root / "lineage.json"
            output_dir = root / "evaluation"
            samples = [
                ("s1", "m1", "Inflation declined to 3 percent in 2024."),
                ("s2", "m1", "Growth rose to 2 percent in 2024."),
                ("s3", "m2", "Employment increased by 1 percent in 2025."),
            ]
            _write_jsonl(
                reference_path,
                [
                    {
                        "sample_id": sample_id,
                        "meeting_id": meeting_id,
                        "section_name": "Economic",
                        "reference": reference,
                    }
                    for sample_id, meeting_id, reference in samples
                ],
            )
            _write_jsonl(
                evidence_path,
                [
                    {
                        "sample_id": sample_id,
                        "meeting_id": meeting_id,
                        "section_name": "Economic",
                        "evidence_text": reference,
                    }
                    for sample_id, meeting_id, reference in samples
                ],
            )
            parents = {
                "base": None,
                "analysis-sft": "base",
                "analysis-grpo": "base",
                "minutes-sft": "analysis-sft",
            }
            generation_rows = []
            for artifact_index, (artifact, parent) in enumerate(
                parents.items(),
                start=1,
            ):
                for sample_id, meeting_id, reference in samples:
                    generation_rows.append(
                        {
                            "artifact_id": artifact,
                            "sample_id": sample_id,
                            "meeting_id": meeting_id,
                            "generated": (
                                reference
                                if artifact != "base"
                                else "A shorter unmatched response."
                            ),
                            "generation_validation_status": "passed",
                            "generation_seed": 20260729,
                            "source_prompt_sha256": _hash_text(sample_id),
                            "provenance": {
                                "checkpoint_sha256": f"{artifact_index:x}" * 64,
                                "parent_artifact_id": parent,
                                "test_set_sha256": "a" * 64,
                                "prompt_template_sha256": "b" * 64,
                                "decoding_config_sha256": "c" * 64,
                            },
                        }
                    )
            _write_jsonl(generation_path, generation_rows)
            lineage_path.write_text(
                json.dumps(
                    {
                        artifact: {"parent": parent}
                        for artifact, parent in parents.items()
                    }
                ),
                encoding="utf-8",
            )
            scorer = InjectedSemanticScorer()

            result = run_checkpoint_generation_evaluation(
                [generation_path],
                reference_path,
                evidence_path,
                output_dir,
                require_formal_manifests=False,
                semantic_scorers=[scorer],
                lineage_manifest_path=lineage_path,
                bootstrap_samples=100,
                bootstrap_seed=20260729,
            )

            self.assertEqual(len(result["row_scores"]), 12)
            self.assertEqual(len(scorer.calls), 1)
            self.assertEqual(result["audit"]["artifact_count"], 4)
            self.assertEqual(result["audit"]["status"], "validated")
            self.assertFalse(result["audit"]["inference_performed"])
            self.assertFalse(result["audit"]["network_required"])
            self.assertTrue(result["audit"]["alignment"]["complete"])
            self.assertTrue(result["audit"]["invalid_output_policy"]["fatal_only"])
            self.assertFalse(
                result["audit"]["format_noncompliance_policy"]["fatal"]
            )
            self.assertEqual(len(result["audit"]["contrast_plan"]), 3)
            for filename in (
                "row_scores.jsonl",
                "claim_checks.jsonl",
                "summary.jsonl",
                "contrasts.jsonl",
                "audit.json",
            ):
                self.assertTrue((output_dir / filename).is_file())
            self.assertEqual(len(_read_jsonl(output_dir / "row_scores.jsonl")), 12)
            contrast_rows_on_disk = _read_jsonl(output_dir / "contrasts.jsonl")
            self.assertTrue(contrast_rows_on_disk)
            self.assertTrue(
                any(row["metric"] == "fake_semantic" for row in contrast_rows_on_disk)
            )
            self.assertTrue(
                all(
                    row["holm_family"] == "all_prespecified_contrast_metric_tests"
                    for row in contrast_rows_on_disk
                )
            )

            reused = run_checkpoint_generation_evaluation(
                [generation_path],
                reference_path,
                evidence_path,
                output_dir,
                require_formal_manifests=False,
                semantic_scorers=[scorer],
                lineage_manifest_path=lineage_path,
                bootstrap_samples=100,
                bootstrap_seed=20260729,
            )
            self.assertTrue(reused["reused_existing"])
            self.assertEqual(len(scorer.calls), 1)

    def test_strict_provenance_rejects_unpinned_generations(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.jsonl"
            evidence = root / "evidence.jsonl"
            generations = root / "generations.jsonl"
            _write_jsonl(
                reference,
                [{"sample_id": "s1", "meeting_id": "m1", "reference": "target"}],
            )
            _write_jsonl(
                evidence,
                [{"sample_id": "s1", "meeting_id": "m1", "evidence_text": "target"}],
            )
            _write_jsonl(
                generations,
                [
                    {
                        "artifact_id": artifact,
                        "sample_id": "s1",
                        "meeting_id": "m1",
                        "generated": "text",
                    }
                    for artifact in ("a", "b", "c", "d")
                ],
            )

            with self.assertRaisesRegex(ValueError, "required provenance"):
                run_checkpoint_generation_evaluation(
                    [generations],
                    reference,
                    evidence,
                    root / "out",
                    require_formal_manifests=False,
                    bootstrap_samples=10,
                )

    def test_distinct_ids_cannot_hide_identical_checkpoint_hashes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.jsonl"
            evidence = root / "evidence.jsonl"
            generations = root / "generations.jsonl"
            _write_jsonl(
                reference,
                [{"sample_id": "s1", "meeting_id": "m1", "reference": "target"}],
            )
            _write_jsonl(
                evidence,
                [{"sample_id": "s1", "meeting_id": "m1", "evidence_text": "target"}],
            )
            _write_jsonl(
                generations,
                [
                    {
                        "artifact_id": artifact,
                        "sample_id": "s1",
                        "meeting_id": "m1",
                        "generated": "target",
                        "provenance": {
                            "checkpoint_sha256": (
                                "1" * 64 if artifact in {"a", "b"} else digit * 64
                            ),
                            "test_set_sha256": "a" * 64,
                            "prompt_template_sha256": "b" * 64,
                            "decoding_config_sha256": "c" * 64,
                        },
                    }
                    for artifact, digit in zip(
                        ("a", "b", "c", "d"),
                        ("1", "2", "3", "4"),
                        strict=True,
                    )
                ],
            )

            with self.assertRaisesRegex(ValueError, "identical checkpoint_sha256"):
                run_checkpoint_generation_evaluation(
                    [generations],
                    reference,
                    evidence,
                    root / "out",
                    require_formal_manifests=False,
                    bootstrap_samples=10,
                )


if __name__ == "__main__":
    unittest.main()
