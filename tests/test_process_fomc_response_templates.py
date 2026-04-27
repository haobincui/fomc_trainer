import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from process_fomc_report.build_qa_master import build_canonical_master
from process_fomc_report.generate_prompt_and_response.algo.build_decision_datasets import (
    build_decision_datasets,
    build_decision_prompts,
)
from process_fomc_report.generate_prompt_and_response.algo.build_minutes_rewrite_dataset import (
    build_minutes_rewrite_dataset,
    build_minutes_rewrite_prompts,
)
from process_fomc_report.generate_prompt_and_response.algo.convert_analysis_sft_to_deepseek_llama import (
    EXPECTED_GENERATION_SUFFIX,
    convert_analysis_sft_to_deepseek_llama,
    convert_gemma_response_to_deepseek_completion,
    load_deepseek_tokenizer,
    render_deepseek_generation_prompt,
)
from process_fomc_report.generate_prompt_and_response.algo.generate_analysis_teacher_responses import (
    generate_teacher_responses,
)
from process_fomc_report.generate_prompt_and_response.algo.generate_minutes_rewrite_teacher_responses import (
    generate_minutes_rewrite_teacher_responses,
)
from process_fomc_report.generate_prompt_and_response.algo.inspect_analysis_sft_resume_state import (
    inspect_analysis_sft_stage_status,
)
from process_fomc_report.generate_prompt_and_response.algo.inspect_decision_resume_state import (
    inspect_decision_stage_status,
)
from process_fomc_report.generate_prompt_and_response.algo.inspect_minutes_alignment_resume_state import (
    inspect_minutes_alignment_stage_status,
)
from process_fomc_report.generate_prompt_and_response.algo.common.response_templates import (
    GEMMA4_THOUGHT_CHANNEL_TEMPLATE,
    LEGACY_XML_TEMPLATE,
    format_response_text,
    parse_response_text,
)
from process_fomc_report.generate_prompt_and_response.algo.normalize_input_sources import (
    normalize_input_sources,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_first_row(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _decision_source_row(meeting_date: str, *, rate_change: str = "No change") -> dict:
    return {
        "prompt": (
            "The current target rate stands at **5.25%** ahead of the "
            f"upcoming meeting on **{meeting_date}**.\n\n"
            "### Allowed Policy Options\n"
            "Raise by 25 basis points\n"
            "No change\n"
            "Cut by 25 basis points\n\n"
            "## Staff Economic and Financial Market Analysis\n"
            "Inflation is moderating while labor market conditions remain firm."
        ),
        "response": rate_change,
        "rate_change": rate_change,
        "provided_data": "table payload",
    }


class TestResponseTemplateHelpers(unittest.TestCase):
    def test_parse_legacy_xml_response(self):
        parsed = parse_response_text("<think>step one</think><answer>final answer</answer>")

        self.assertEqual(parsed.format_name, LEGACY_XML_TEMPLATE)
        self.assertEqual(parsed.reasoning, "step one")
        self.assertEqual(parsed.answer, "final answer")

    def test_convert_legacy_xml_to_gemma_thought_channel(self):
        converted = format_response_text(
            "<think>step one</think><answer>final answer</answer>",
            response_template=GEMMA4_THOUGHT_CHANNEL_TEMPLATE,
        )

        self.assertEqual(
            converted,
            "<|channel>thought\nstep one\n<channel|>final answer",
        )

    def test_gemma_thought_channel_is_idempotent(self):
        response = "<|channel>thought\nstep one\n<channel|>final answer"

        converted = format_response_text(
            response,
            response_template=GEMMA4_THOUGHT_CHANNEL_TEMPLATE,
        )

        self.assertEqual(converted, response)

    def test_plain_answer_remains_plain_in_both_modes(self):
        answer = "final answer only"

        self.assertEqual(
            format_response_text(answer, response_template=LEGACY_XML_TEMPLATE),
            answer,
        )
        self.assertEqual(
            format_response_text(
                answer,
                response_template=GEMMA4_THOUGHT_CHANNEL_TEMPLATE,
            ),
            answer,
        )

    def test_reasoning_override_builds_structured_response(self):
        converted = format_response_text(
            "final answer",
            response_template=GEMMA4_THOUGHT_CHANNEL_TEMPLATE,
            reasoning_text="step one",
        )

        self.assertEqual(
            converted,
            "<|channel>thought\nstep one\n<channel|>final answer",
        )

    def test_empty_legacy_xml_normalizes_to_empty_text_in_gemma_mode(self):
        converted = format_response_text(
            "<think></think><answer></answer>",
            response_template=GEMMA4_THOUGHT_CHANNEL_TEMPLATE,
        )

        self.assertEqual(converted, "")


class _FakeDeepSeekTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if tokenize:
            raise AssertionError("Tests expect non-tokenized chat template rendering.")
        rendered = "<｜begin▁of▁sentence｜>"
        for message in messages:
            role = message["role"]
            if role == "system":
                rendered += message["content"]
            elif role == "user":
                rendered += "<｜User｜>" + message["content"]
            else:
                raise AssertionError(f"Unexpected role: {role}")
        if add_generation_prompt:
            rendered += EXPECTED_GENERATION_SUFFIX
        return rendered


class TestConvertAnalysisSftToDeepSeekLlama(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.local_model_path = cls.repo_root / "models" / "DeepSeek-R1-Distill-Llama-8B"

    def test_convert_gemma_response_to_deepseek_completion(self):
        converted = convert_gemma_response_to_deepseek_completion(
            "<|channel>thought\nstep one\nstep two\n<channel|>final answer"
        )

        self.assertEqual(converted, "step one\nstep two\n</think>\nfinal answer")

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "models" / "DeepSeek-R1-Distill-Llama-8B").exists(),
        "Local DeepSeek-R1-Distill-Llama-8B model is unavailable.",
    )
    def test_local_deepseek_tokenizer_generation_prompt_ends_with_think(self):
        tokenizer = load_deepseek_tokenizer(self.local_model_path)
        rendered_prompt = render_deepseek_generation_prompt(tokenizer)

        self.assertTrue(rendered_prompt.endswith(EXPECTED_GENERATION_SUFFIX))

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "models" / "DeepSeek-R1-Distill-Llama-8B").exists(),
        "Local DeepSeek-R1-Distill-Llama-8B model is unavailable.",
    )
    def test_local_prompt_plus_converted_completion_has_expected_boundary(self):
        tokenizer = load_deepseek_tokenizer(self.local_model_path)
        rendered_prompt = render_deepseek_generation_prompt(tokenizer)
        completion = convert_gemma_response_to_deepseek_completion(
            "<|channel>thought\nreasoning text\n<channel|>answer text"
        )

        self.assertEqual(
            rendered_prompt + completion,
            f"{rendered_prompt}reasoning text\n</think>\nanswer text",
        )
        self.assertIn("<｜Assistant｜><think>\nreasoning text\n</think>\nanswer text", rendered_prompt + completion)

    @patch(
        "process_fomc_report.generate_prompt_and_response.algo.convert_analysis_sft_to_deepseek_llama.load_deepseek_tokenizer",
        return_value=_FakeDeepSeekTokenizer(),
    )
    def test_end_to_end_conversion_writes_expected_files(self, _mock_tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source"
            output_root = root / "output"
            minimal_row = {
                "prompt": "prompt",
                "response": "<|channel>thought\nreasoning\n<channel|>answer",
                "provided_data": "table",
            }
            manifest_row = {
                "sample_id": "sample-1",
                "meeting_date": "2024-01-31",
                "prompt": "prompt",
                "response": "<|channel>thought\nreasoning\n<channel|>answer",
                "provided_data": "table",
                "topic": "Federal Funds Rate",
            }

            for split in ("train", "eval", "test"):
                _write_jsonl(source_root / f"{split}.jsonl", [minimal_row])
                _write_jsonl(source_root / f"{split}_manifest.jsonl", [manifest_row])

            summary = convert_analysis_sft_to_deepseek_llama(
                model_path="unused",
                source_root=source_root,
                output_root=output_root,
            )

            self.assertEqual(summary["splits"]["train"]["rows_read"], 1)
            self.assertEqual(summary["splits"]["eval"]["rows_written"], 1)
            self.assertEqual(summary["splits"]["test"]["parse_failures"], 0)

            expected_files = {
                output_root / "train.jsonl",
                output_root / "eval.jsonl",
                output_root / "test.jsonl",
                output_root / "train_manifest.jsonl",
                output_root / "eval_manifest.jsonl",
                output_root / "test_manifest.jsonl",
            }
            for path in expected_files:
                self.assertTrue(path.exists(), msg=f"Missing output file: {path}")

            train_rows = _load_jsonl(output_root / "train.jsonl")
            manifest_rows = _load_jsonl(output_root / "train_manifest.jsonl")
            self.assertEqual(len(train_rows), 1)
            self.assertEqual(len(manifest_rows), 1)
            self.assertEqual(train_rows[0].keys(), {"prompt", "response", "provided_data"})
            self.assertEqual(train_rows[0]["response"], "reasoning\n</think>\nanswer")
            self.assertEqual(manifest_rows[0]["response"], "reasoning\n</think>\nanswer")
            self.assertEqual(manifest_rows[0]["sample_id"], "sample-1")

    @patch(
        "process_fomc_report.generate_prompt_and_response.algo.convert_analysis_sft_to_deepseek_llama.load_deepseek_tokenizer",
        return_value=_FakeDeepSeekTokenizer(),
    )
    def test_malformed_response_fails_fast(self, _mock_tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source"
            output_root = root / "output"

            _write_jsonl(
                source_root / "train.jsonl",
                [{"prompt": "prompt", "response": "plain answer", "provided_data": "table"}],
            )
            _write_jsonl(
                source_root / "train_manifest.jsonl",
                [{"prompt": "prompt", "response": "plain answer", "provided_data": "table"}],
            )
            for split in ("eval", "test"):
                _write_jsonl(
                    source_root / f"{split}.jsonl",
                    [{"prompt": "prompt", "response": "<|channel>thought\nreasoning\n<channel|>answer", "provided_data": "table"}],
                )
                _write_jsonl(
                    source_root / f"{split}_manifest.jsonl",
                    [{"prompt": "prompt", "response": "<|channel>thought\nreasoning\n<channel|>answer", "provided_data": "table"}],
                )

            with self.assertRaisesRegex(ValueError, r"Failed converting training row 1 for split=train"):
                convert_analysis_sft_to_deepseek_llama(
                    model_path="unused",
                    source_root=source_root,
                    output_root=output_root,
                )


class TestNormalizeInputSources(unittest.TestCase):
    def test_normalize_input_sources_respects_configured_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy_response = "<think>step one</think><answer>final answer</answer>"
            expected_response = "<|channel>thought\nstep one\n<channel|>final answer"

            fomc_qa = root / "fomc_qa.jsonl"
            decision_sft_train = root / "decision_sft_train.jsonl"
            decision_sft_eval = root / "decision_sft_eval.jsonl"
            decision_sft_test = root / "decision_sft_test.jsonl"
            decision_grpo_dir = root / "decision_grpo"
            decision_grpo_train = decision_grpo_dir / "decision_grpo_train.jsonl"

            for path in [
                fomc_qa,
                decision_sft_train,
                decision_sft_eval,
                decision_sft_test,
                decision_grpo_train,
            ]:
                _write_jsonl(path, [{"prompt": "p", "response": legacy_response, "provided_data": ""}])

            config_path = root / "prompt_pipeline.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "response_template": GEMMA4_THOUGHT_CHANNEL_TEMPLATE,
                        "pipeline": {
                            "input_root": str(root),
                            "audit_root": str(root / "audit"),
                        },
                        "decision": {
                            "sft_source_files": {
                                "train": str(decision_sft_train),
                                "eval": str(decision_sft_eval),
                                "test": str(decision_sft_test),
                            },
                            "grpo_source_pattern": str(decision_grpo_dir / "*.jsonl"),
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            summary = normalize_input_sources(str(config_path))
            self.assertEqual(summary["response_template"], GEMMA4_THOUGHT_CHANNEL_TEMPLATE)
            self.assertTrue(any(item["changed_rows"] == 1 for item in summary["files"].values()))
            self.assertTrue(Path(summary["audit_path"]).exists())
            self.assertEqual(summary["marker"]["files"][str(fomc_qa)]["rows"], 1)

            self.assertEqual(_load_first_row(fomc_qa)["response"], expected_response)
            self.assertEqual(_load_first_row(decision_sft_train)["response"], expected_response)
            self.assertEqual(_load_first_row(decision_grpo_train)["response"], expected_response)

            rerun_summary = normalize_input_sources(str(config_path))
            self.assertTrue(all(item["changed_rows"] == 0 for item in rerun_summary["files"].values()))


class TestAnalysisSftResumeInspector(unittest.TestCase):
    def test_inspector_marks_partial_chk1_progress_for_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "input_sources"
            pipeline_root = root / "pipeline" / "analysis_sft"
            audit_root = root / "pipeline" / "audit"
            train_root = root / "train" / "analysis_sft"
            prompt_root = pipeline_root / "teacher_prompts" / "after_2009"
            teacher_response_root = pipeline_root / "teacher_responses" / "after_2009"
            student_prompt_root = pipeline_root / "student_prompts" / "after_2009"
            manifest_root = root / "manifests"
            decision_sft_root = input_root / "decision_making"
            decision_grpo_root = input_root / "decision_grpo"
            merged_labeled_path = pipeline_root / "labeled" / "merged_labeled_after_2009.xlsx"
            config_path = root / "prompt_pipeline.yaml"

            split_counts = {"train": 2, "eval": 1, "test": 1}
            qa_rows = [
                {"prompt": f"qa-prompt-{idx}", "response": f"qa-response-{idx}", "provided_data": ""}
                for idx in range(4)
            ]
            meeting_dates = ["2024-01-31", "2024-02-01", "2024-02-02", "2024-02-03"]
            manifest_rows = {
                "train": [
                    {
                        "sample_id": "sample-train-1",
                        "meeting_date": meeting_dates[0],
                        "prompt": "teacher prompt train 1",
                        "response": "teacher response train 1",
                        "provided_data": "table train 1",
                        "prompt_hash": "teacher-hash-train-1",
                    },
                    {
                        "sample_id": "sample-train-2",
                        "meeting_date": meeting_dates[1],
                        "prompt": "teacher prompt train 2",
                        "response": "teacher response train 2",
                        "provided_data": "table train 2",
                        "prompt_hash": "teacher-hash-train-2",
                    },
                ],
                "eval": [
                    {
                        "sample_id": "sample-eval-1",
                        "meeting_date": meeting_dates[2],
                        "prompt": "teacher prompt eval 1",
                        "response": "teacher response eval 1",
                        "provided_data": "table eval 1",
                        "prompt_hash": "teacher-hash-eval-1",
                    }
                ],
                "test": [
                    {
                        "sample_id": "sample-test-1",
                        "meeting_date": meeting_dates[3],
                        "prompt": "teacher prompt test 1",
                        "response": "teacher response test 1",
                        "provided_data": "table test 1",
                        "prompt_hash": "teacher-hash-test-1",
                    }
                ],
            }

            _write_jsonl(pipeline_root / "master" / "qa_master.jsonl", qa_rows)
            _write_jsonl(input_root / "fomc_qa.jsonl", qa_rows)
            for split, rows in manifest_rows.items():
                _write_jsonl(manifest_root / f"qa_{split}_manifest.jsonl", rows)
                _write_jsonl(pipeline_root / "meeting_level" / f"{split}.jsonl", rows)
                _write_jsonl(pipeline_root / "sample_level_legacy" / f"{split}.jsonl", rows)

            for audit_name in (
                "source_reconciliation.json",
                "backfilled_rows.json",
                "dropped_rows.json",
                "meeting_split_manifest.json",
                "legacy_split_manifest.json",
            ):
                audit_path = pipeline_root / "audit" / audit_name
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                audit_path.write_text("{}", encoding="utf-8")

            merged_labeled_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"value": list(range(6))}).to_excel(merged_labeled_path, index=False)

            teacher_prompt_rows = {
                "train": [
                    {
                        "sample_id": "sample-train-1",
                        "prompt": "teacher prompt train 1",
                        "prompt_hash": "teacher-hash-train-1",
                    },
                    {
                        "sample_id": "sample-train-2",
                        "prompt": "teacher prompt train 2",
                        "prompt_hash": "teacher-hash-train-2",
                    },
                ],
                "eval": [
                    {
                        "sample_id": "sample-eval-1",
                        "prompt": "teacher prompt eval 1",
                        "prompt_hash": "teacher-hash-eval-1",
                    }
                ],
                "test": [
                    {
                        "sample_id": "sample-test-1",
                        "prompt": "teacher prompt test 1",
                        "prompt_hash": "teacher-hash-test-1",
                    }
                ],
            }
            for split, rows in teacher_prompt_rows.items():
                _write_jsonl(prompt_root / f"{split}.jsonl", rows)

            _write_jsonl(
                teacher_response_root / "train.jsonl",
                [
                    {
                        "sample_id": "sample-train-1",
                        "split": "train",
                        "prompt_hash": "teacher-hash-train-1",
                        "response": "<think>reason</think><answer>answer</answer>",
                        "reasoning": "reason",
                        "teacher_model": "teacher-model",
                        "status": "success",
                        "response_template": LEGACY_XML_TEMPLATE,
                    }
                ],
            )
            _write_jsonl(
                teacher_response_root / "eval.jsonl",
                [
                    {
                        "sample_id": "sample-eval-1",
                        "split": "eval",
                        "prompt_hash": "teacher-hash-eval-1",
                        "response": "<think>reason</think><answer>answer</answer>",
                        "reasoning": "reason",
                        "teacher_model": "teacher-model",
                        "status": "success",
                        "response_template": LEGACY_XML_TEMPLATE,
                    }
                ],
            )
            _write_jsonl(
                teacher_response_root / "test.jsonl",
                [
                    {
                        "sample_id": "sample-test-1",
                        "split": "test",
                        "prompt_hash": "teacher-hash-test-1",
                        "response": "<think>reason</think><answer>answer</answer>",
                        "reasoning": "reason",
                        "teacher_model": "teacher-model",
                        "status": "success",
                        "response_template": LEGACY_XML_TEMPLATE,
                    }
                ],
            )

            for path in (
                decision_sft_root / "decision_train.jsonl",
                decision_sft_root / "decision_eval.jsonl",
                decision_sft_root / "decision_test.jsonl",
                decision_grpo_root / "decision_grpo_train.jsonl",
            ):
                _write_jsonl(path, [{"prompt": "decision prompt", "response": "decision response", "provided_data": ""}])

            config_path.write_text(
                yaml.safe_dump(
                    {
                        "response_template": LEGACY_XML_TEMPLATE,
                        "pipeline": {
                            "input_root": str(input_root),
                            "audit_root": str(audit_root),
                        },
                        "analysis": {
                            "pipeline_root": str(pipeline_root),
                            "master_path": str(pipeline_root / "master" / "qa_master.jsonl"),
                            "labeled_paths": {"after_2009": str(merged_labeled_path)},
                            "split_manifests": {
                                "after_2009": {
                                    split: str(manifest_root / f"qa_{split}_manifest.jsonl")
                                    for split in ("train", "eval", "test")
                                }
                            },
                            "teacher_prompt_root": str(prompt_root.parent),
                            "teacher_response_root": str(teacher_response_root.parent),
                            "analysis_sft_prompt_root": str(student_prompt_root.parent),
                            "analysis_grpo_prompt_root": str(root / "unused_grpo_prompts"),
                            "train_roots": {
                                "analysis_sft": str(train_root),
                                "analysis_grpo": str(root / "unused_grpo_train"),
                            },
                            "compat_sft_train_rows": 1,
                            "compat_grpo_core_rows": 1,
                            "replay_stride": 10,
                        },
                        "profiles": {
                            "compat": {
                                "require_nonempty_table": False,
                                "drop_abnormal_section": False,
                                "drop_abnormal_topic": False,
                                "drop_missing_reference": False,
                                "drop_too_long_prompt": False,
                                "drop_missing_indicator_data": False,
                            },
                            "strict": {
                                "require_nonempty_table": False,
                                "drop_abnormal_section": False,
                                "drop_abnormal_topic": False,
                                "drop_missing_reference": False,
                                "drop_too_long_prompt": False,
                                "drop_missing_indicator_data": False,
                            },
                        },
                        "decision": {
                            "sft_source_files": {
                                "train": str(decision_sft_root / "decision_train.jsonl"),
                                "eval": str(decision_sft_root / "decision_eval.jsonl"),
                                "test": str(decision_sft_root / "decision_test.jsonl"),
                            },
                            "grpo_source_pattern": str(decision_grpo_root / "*.jsonl"),
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            normalize_input_sources(str(config_path))

            status = inspect_analysis_sft_stage_status(
                str(config_path),
                profile="compat",
                scope="after_2009",
                expected_split_counts=split_counts,
                expected_qa_master_rows=4,
                expected_labeled_rows=6,
            )

            self.assertTrue(status["stages"]["build_qa_master"]["complete"])
            self.assertTrue(status["stages"]["normalize_input_sources"]["complete"])
            self.assertTrue(status["stages"]["merge_labels"]["complete"])
            self.assertTrue(status["stages"]["analysis_sft_teacher_prompts"]["complete"])
            self.assertFalse(status["stages"]["analysis_sft_teacher_responses"]["complete"])
            self.assertFalse(status["stages"]["analysis_sft_prompts"]["complete"])
            self.assertFalse(status["stages"]["analysis_sft_dataset"]["complete"])


class TestDecisionResumeInspector(unittest.TestCase):
    def test_inspector_marks_partial_chk4_progress_for_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "input_sources"
            prompt_root = root / "pipeline" / "decision"
            train_root = root / "train"
            config_path = root / "prompt_pipeline.yaml"

            _write_jsonl(
                source_root / "decision_making" / "train.jsonl",
                [
                    _decision_source_row("2024-01-31"),
                    _decision_source_row("2024-03-20", rate_change="Raise by 25 basis points"),
                ],
            )
            _write_jsonl(
                source_root / "decision_making" / "eval.jsonl",
                [_decision_source_row("2024-05-01")],
            )
            _write_jsonl(
                source_root / "decision_making" / "test.jsonl",
                [_decision_source_row("2024-06-12", rate_change="Cut by 25 basis points")],
            )
            _write_jsonl(
                source_root / "decision_grpo" / "grpo.jsonl",
                [
                    _decision_source_row("2024-01-31"),
                    _decision_source_row("2024-03-20"),
                    _decision_source_row("2024-05-01"),
                ],
            )

            config_path.write_text(
                yaml.safe_dump(
                    {
                        "response_template": GEMMA4_THOUGHT_CHANNEL_TEMPLATE,
                        "pipeline": {"audit_root": str(root / "audit")},
                        "decision": {
                            "sft_source_files": {
                                "train": str(source_root / "decision_making" / "train.jsonl"),
                                "eval": str(source_root / "decision_making" / "eval.jsonl"),
                                "test": str(source_root / "decision_making" / "test.jsonl"),
                            },
                            "grpo_source_pattern": str(source_root / "decision_grpo" / "*.jsonl"),
                            "prompt_roots": {
                                "decision_sft": str(prompt_root / "decision_sft_prompts"),
                                "decision_grpo": str(prompt_root / "decision_grpo_prompts"),
                            },
                            "train_roots": {
                                "decision_sft": str(train_root / "decision_sft"),
                                "decision_grpo": str(train_root / "decision_grpo"),
                            },
                            "split_seed": 42,
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            build_decision_prompts(str(config_path), scope="after_2009")
            partial_status = inspect_decision_stage_status(str(config_path), scope="after_2009")

            self.assertTrue(partial_status["stages"]["decision_prompts"]["complete"])
            self.assertFalse(partial_status["stages"]["decision_dataset"]["complete"])

            build_decision_datasets(str(config_path), scope="after_2009")
            complete_status = inspect_decision_stage_status(str(config_path), scope="after_2009")

            self.assertTrue(complete_status["stages"]["decision_prompts"]["complete"])
            self.assertTrue(complete_status["stages"]["decision_dataset"]["complete"])

            _write_jsonl(train_root / "decision_sft" / "train_manifest.jsonl", [])
            stale_status = inspect_decision_stage_status(str(config_path), scope="after_2009")

            self.assertFalse(stale_status["stages"]["decision_dataset"]["complete"])
            self.assertIn(
                "decision_sft_train_manifest_count_mismatch",
                stale_status["stages"]["decision_dataset"]["reason"],
            )


class TestBuildQaMasterRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy_master_rows, cls.legacy_audit = build_canonical_master(
            response_template=LEGACY_XML_TEMPLATE
        )
        cls.gemma_master_rows, cls.gemma_audit = build_canonical_master(
            response_template=GEMMA4_THOUGHT_CHANNEL_TEMPLATE
        )

    def test_build_canonical_master_preserves_legacy_counts(self):
        self.assertEqual(len(self.legacy_master_rows), 4887)
        self.assertEqual(len(self.legacy_audit["backfilled_rows"]), 2)
        self.assertEqual(len(self.legacy_audit["dropped_rows"]), 1)
        self.assertEqual(
            self.legacy_audit["source_reconciliation"]["canonical_counts"]["empty_responses"],
            0,
        )

    def test_build_canonical_master_preserves_counts_for_gemma_mode(self):
        self.assertEqual(len(self.gemma_master_rows), 4887)
        self.assertEqual(len(self.gemma_audit["backfilled_rows"]), 2)
        self.assertEqual(len(self.gemma_audit["dropped_rows"]), 1)
        backfilled_rows = [
            row for row in self.gemma_master_rows if row["response_origin"] == "backfilled_from_dataset_raw"
        ]
        self.assertEqual(len(backfilled_rows), len(self.gemma_audit["backfilled_rows"]))
        self.assertTrue(all(row["response"] == "" for row in backfilled_rows))


class TestGenerateTeacherResponses(unittest.TestCase):
    def test_force_refresh_ignores_existing_teacher_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "prompt_pipeline.yaml"
            master_path = root / "analysis_sft" / "master" / "qa_master.jsonl"
            prompt_root = root / "analysis_sft" / "teacher_prompts" / "after_2009"
            response_root = root / "analysis_sft" / "teacher_responses" / "after_2009"

            _write_jsonl(
                master_path,
                [
                    {
                        "sample_id": "sample-train",
                        "prompt": "p",
                        "response": "old archived",
                    },
                    {
                        "sample_id": "sample-eval",
                        "prompt": "p",
                        "response": "old archived",
                    },
                    {
                        "sample_id": "sample-test",
                        "prompt": "p",
                        "response": "old archived",
                    },
                ],
            )
            for split in ("train", "eval", "test"):
                _write_jsonl(
                    prompt_root / f"{split}.jsonl",
                    [
                        {
                            "sample_id": f"sample-{split}",
                            "prompt": f"prompt-{split}",
                            "prompt_hash": f"hash-{split}",
                        }
                    ],
                )
                _write_jsonl(
                    response_root / f"{split}.jsonl",
                    [
                        {
                            "sample_id": f"sample-{split}",
                            "split": split,
                            "prompt_hash": f"hash-{split}",
                            "response": "cached response",
                            "reasoning": "cached reasoning",
                            "teacher_model": "cached-model",
                            "status": "success",
                            "response_template": LEGACY_XML_TEMPLATE,
                        }
                    ],
                )

            config_path.write_text(
                yaml.safe_dump(
                    {
                        "response_template": LEGACY_XML_TEMPLATE,
                        "analysis": {
                            "master_path": str(master_path),
                            "teacher_prompt_root": str(prompt_root.parent),
                            "teacher_response_root": str(response_root.parent),
                        },
                        "pipeline": {
                            "audit_root": str(root / "audit"),
                        },
                        "teacher": {
                            "source": "llm",
                            "model_name": "test-model",
                            "api_key": "test-key",
                            "resume": True,
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with patch(
                "process_fomc_report.generate_prompt_and_response.algo.generate_analysis_teacher_responses.get_response",
                return_value=("fresh answer", "fresh reasoning"),
            ) as mocked_get_response:
                generate_teacher_responses(
                    str(config_path),
                    scope="after_2009",
                    teacher_source="llm",
                    force_refresh=True,
                )

            self.assertEqual(mocked_get_response.call_count, 3)
            self.assertEqual(
                _load_first_row(response_root / "train.jsonl")["response"],
                "<think>fresh reasoning</think><answer>fresh answer</answer>",
            )
            self.assertEqual(_load_first_row(response_root / "train.jsonl")["reasoning"], "fresh reasoning")


class TestMinutesAlignmentPipeline(unittest.TestCase):
    def test_minutes_alignment_resume_inspector_tracks_partial_chk3_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            analysis_prompt_root = root / "analysis_sft" / "teacher_prompts" / "after_2009"
            analysis_teacher_root = root / "analysis_sft" / "teacher_responses" / "after_2009"
            prompt_root = root / "minutes_alignment" / "prompts"
            response_root = root / "minutes_alignment" / "teacher_responses" / "after_2009"
            target_root = root / "minutes_alignment" / "targets"
            train_root = root / "minutes_alignment" / "train"
            config_path = root / "prompt_pipeline.yaml"

            for split in ("train", "eval", "test"):
                sample_id = f"sample-{split}"
                prompt_hash_value = f"analysis-hash-{split}"
                _write_jsonl(
                    analysis_prompt_root / f"{split}.jsonl",
                    [
                        {
                            "sample_id": sample_id,
                            "split": split,
                            "meeting_date": "2024-01-31",
                            "section_name": "Inflation",
                            "topic": "prices",
                            "source_row_index": 7,
                            "prompt_hash": prompt_hash_value,
                            "reference_excerpt": f"Reference excerpt {split}",
                            "provided_data": "table payload",
                            "quality_flags": [],
                        }
                    ],
                )
                _write_jsonl(
                    analysis_teacher_root / f"{split}.jsonl",
                    [
                        {
                            "sample_id": sample_id,
                            "split": split,
                            "prompt_hash": prompt_hash_value,
                            "response": "<think>analysis reasoning</think><answer>Teacher analysis</answer>",
                            "reasoning": "analysis reasoning",
                            "teacher_model": "analysis-model",
                            "status": "success",
                            "response_template": LEGACY_XML_TEMPLATE,
                        }
                    ],
                )

            config_path.write_text(
                yaml.safe_dump(
                    {
                        "response_template": LEGACY_XML_TEMPLATE,
                        "rewrite": {
                            "source_prompt_root": str(analysis_prompt_root.parent),
                            "source_teacher_response_root": str(analysis_teacher_root.parent),
                            "prompt_root": str(prompt_root),
                            "teacher_response_root": str(response_root.parent),
                            "normalized_target_root": str(target_root),
                            "train_root": str(train_root),
                        },
                        "pipeline": {"audit_root": str(root / "audit")},
                        "teacher": {
                            "model_name": "rewrite-model",
                            "api_key": "test-key",
                            "resume": True,
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            build_minutes_rewrite_prompts(str(config_path), scope="after_2009")
            partial_status = inspect_minutes_alignment_stage_status(str(config_path), scope="after_2009")

            self.assertTrue(partial_status["stages"]["minutes_alignment_prompts"]["complete"])
            self.assertFalse(partial_status["stages"]["minutes_alignment_teacher_responses"]["complete"])
            self.assertFalse(partial_status["stages"]["minutes_alignment_dataset"]["complete"])

            for split in ("train", "eval", "test"):
                prompt_row = _load_first_row(prompt_root / "after_2009" / f"{split}.jsonl")
                _write_jsonl(
                    response_root / f"{split}.jsonl",
                    [
                        {
                            "sample_id": prompt_row["sample_id"],
                            "split": split,
                            "prompt_hash": prompt_row["prompt_hash"],
                            "response": "<think>rewrite reasoning</think><answer>Rewritten minutes</answer>",
                            "reasoning": "rewrite reasoning",
                            "teacher_model": "rewrite-model",
                            "status": "success",
                            "response_template": LEGACY_XML_TEMPLATE,
                        }
                    ],
                )

            teacher_status = inspect_minutes_alignment_stage_status(str(config_path), scope="after_2009")

            self.assertTrue(teacher_status["stages"]["minutes_alignment_prompts"]["complete"])
            self.assertTrue(teacher_status["stages"]["minutes_alignment_teacher_responses"]["complete"])
            self.assertFalse(teacher_status["stages"]["minutes_alignment_dataset"]["complete"])

            build_minutes_rewrite_dataset(str(config_path), scope="after_2009")
            complete_status = inspect_minutes_alignment_stage_status(str(config_path), scope="after_2009")

            self.assertTrue(complete_status["stages"]["minutes_alignment_prompts"]["complete"])
            self.assertTrue(complete_status["stages"]["minutes_alignment_teacher_responses"]["complete"])
            self.assertTrue(complete_status["stages"]["minutes_alignment_dataset"]["complete"])

            _write_jsonl(train_root / "train_manifest.jsonl", [])
            stale_status = inspect_minutes_alignment_stage_status(str(config_path), scope="after_2009")

            self.assertFalse(stale_status["stages"]["minutes_alignment_dataset"]["complete"])
            self.assertIn(
                "train_manifest_count_mismatch",
                stale_status["stages"]["minutes_alignment_dataset"]["reason"],
            )

    def test_minutes_alignment_prompts_use_analysis_teacher_answer_as_raw_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_root = root / "analysis_sft" / "teacher_prompts" / "after_2009"
            teacher_root = root / "analysis_sft" / "teacher_responses" / "after_2009"
            rewrite_prompt_root = root / "minutes_alignment" / "prompts"
            config_path = root / "prompt_pipeline.yaml"

            _write_jsonl(
                prompt_root / "train.jsonl",
                [
                    {
                        "sample_id": "sample-train",
                        "split": "train",
                        "meeting_date": "2024-01-31",
                        "section_name": "Inflation",
                        "topic": "prices",
                        "source_row_index": 7,
                        "prompt_hash": "analysis-hash",
                        "reference_excerpt": "Reference excerpt",
                        "provided_data": "table payload",
                        "quality_flags": [],
                    }
                ],
            )
            _write_jsonl(prompt_root / "eval.jsonl", [])
            _write_jsonl(prompt_root / "test.jsonl", [])
            _write_jsonl(
                teacher_root / "train.jsonl",
                [
                    {
                        "sample_id": "sample-train",
                        "split": "train",
                        "prompt_hash": "analysis-hash",
                        "response": "<|channel>thought\nteacher analysis reasoning\n<channel|>Teacher final analysis paragraph",
                        "reasoning": "teacher analysis reasoning",
                        "teacher_model": "analysis-model",
                        "status": "success",
                        "response_template": GEMMA4_THOUGHT_CHANNEL_TEMPLATE,
                    }
                ],
            )
            _write_jsonl(teacher_root / "eval.jsonl", [])
            _write_jsonl(teacher_root / "test.jsonl", [])

            config_path.write_text(
                yaml.safe_dump(
                    {
                        "rewrite": {
                            "source_prompt_root": str(prompt_root.parent),
                            "source_teacher_response_root": str(teacher_root.parent),
                            "prompt_root": str(rewrite_prompt_root),
                        },
                        "pipeline": {"audit_root": str(root / "audit")},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            build_minutes_rewrite_prompts(str(config_path), scope="after_2009")

            row = _load_first_row(rewrite_prompt_root / "after_2009" / "train.jsonl")
            self.assertEqual(row["sample_id"], "sample-train")
            self.assertEqual(row["raw_analysis"], "Teacher final analysis paragraph")
            self.assertEqual(row["reference_excerpt"], "Reference excerpt")
            self.assertEqual(row["provided_data"], "table payload")

    def test_minutes_alignment_teacher_responses_resume_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_root = root / "minutes_alignment" / "prompts" / "after_2009"
            response_root = root / "minutes_alignment" / "teacher_responses" / "after_2009"
            config_path = root / "prompt_pipeline.yaml"

            for split in ("train", "eval", "test"):
                _write_jsonl(
                    prompt_root / f"{split}.jsonl",
                    [
                        {
                            "sample_id": f"sample-{split}",
                            "prompt": f"rewrite prompt {split}",
                            "prompt_hash": f"rewrite-hash-{split}",
                        }
                    ],
                )
                _write_jsonl(
                    response_root / f"{split}.jsonl",
                    [
                        {
                            "sample_id": f"sample-{split}",
                            "split": split,
                            "prompt_hash": f"rewrite-hash-{split}",
                            "response": "cached rewrite response",
                            "reasoning": "cached rewrite reasoning",
                            "teacher_model": "cached-model",
                            "status": "success",
                            "response_template": LEGACY_XML_TEMPLATE,
                        }
                    ],
                )

            config_path.write_text(
                yaml.safe_dump(
                    {
                        "response_template": LEGACY_XML_TEMPLATE,
                        "rewrite": {
                            "prompt_root": str(prompt_root.parent),
                            "teacher_response_root": str(response_root.parent),
                        },
                        "pipeline": {"audit_root": str(root / "audit")},
                        "teacher": {
                            "model_name": "rewrite-model",
                            "api_key": "test-key",
                            "resume": True,
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with patch(
                "process_fomc_report.generate_prompt_and_response.algo.generate_minutes_rewrite_teacher_responses.get_response",
                side_effect=AssertionError("cached rows should not call llm"),
            ):
                generate_minutes_rewrite_teacher_responses(str(config_path), scope="after_2009")

            self.assertEqual(_load_first_row(response_root / "train.jsonl")["response"], "cached rewrite response")

    def test_minutes_alignment_dataset_uses_rewrite_reasoning_and_reference_excerpt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_root = root / "minutes_alignment" / "prompts" / "after_2009"
            teacher_root = root / "minutes_alignment" / "teacher_responses" / "after_2009"
            target_root = root / "minutes_alignment" / "targets"
            train_root = root / "minutes_alignment" / "train"
            config_path = root / "prompt_pipeline.yaml"

            _write_jsonl(
                prompt_root / "train.jsonl",
                [
                    {
                        "sample_id": "sample-train",
                        "split": "train",
                        "meeting_date": "2024-01-31",
                        "section_name": "Inflation",
                        "source_row_index": 7,
                        "quality_flags": [],
                        "prompt_hash": "rewrite-hash",
                        "prompt": "rewrite prompt",
                        "provided_data": "table payload",
                        "reference_excerpt": "Canonical reference excerpt",
                        "raw_analysis": "Teacher final analysis paragraph",
                    }
                ],
            )
            _write_jsonl(prompt_root / "eval.jsonl", [])
            _write_jsonl(prompt_root / "test.jsonl", [])
            _write_jsonl(
                teacher_root / "train.jsonl",
                [
                    {
                        "sample_id": "sample-train",
                        "split": "train",
                        "prompt_hash": "rewrite-hash",
                        "response": "<|channel>thought\nrewrite reasoning\n<channel|>Rewritten minutes paragraph",
                        "reasoning": "rewrite reasoning",
                        "teacher_model": "rewrite-model",
                        "status": "success",
                        "response_template": GEMMA4_THOUGHT_CHANNEL_TEMPLATE,
                    }
                ],
            )
            _write_jsonl(teacher_root / "eval.jsonl", [])
            _write_jsonl(teacher_root / "test.jsonl", [])

            config_path.write_text(
                yaml.safe_dump(
                    {
                        "response_template": GEMMA4_THOUGHT_CHANNEL_TEMPLATE,
                        "rewrite": {
                            "prompt_root": str(prompt_root.parent),
                            "teacher_response_root": str(teacher_root.parent),
                            "normalized_target_root": str(target_root),
                            "train_root": str(train_root),
                        },
                        "pipeline": {"audit_root": str(root / "audit")},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            build_minutes_rewrite_dataset(str(config_path), scope="after_2009")

            train_row = _load_first_row(train_root / "train.jsonl")
            manifest_row = _load_first_row(train_root / "train_manifest.jsonl")
            target_row = _load_first_row(target_root / "train.jsonl")
            expected_response = "<|channel>thought\nrewrite reasoning\n<channel|>Canonical reference excerpt"

            self.assertEqual(train_row["response"], expected_response)
            self.assertEqual(manifest_row["teacher_rewrite_response"], "Rewritten minutes paragraph")
            self.assertEqual(manifest_row["teacher_rewrite_reasoning"], "rewrite reasoning")
            self.assertEqual(target_row["response"], expected_response)


if __name__ == "__main__":
    unittest.main()
