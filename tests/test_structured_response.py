import unittest

from open_r1.structured_response import (
    DEEPSEEK_THINK_COMPLETION_FORMAT,
    LENGTH_TOLERANT_ANSWER_TAG_MODE,
    LENGTH_TOLERANT_EMPTY_MODE,
    LENGTH_TOLERANT_FULL_COMPLETION_MODE,
    extract_answer,
    extract_length_tolerant_candidate,
    parse_structured_response,
)


class TestDeepSeekClosingBoundary(unittest.TestCase):
    def test_optional_opening_think_tag_does_not_require_answer_wrapper(self):
        parsed = parse_structured_response(
            "<think>Evidence-window reasoning.</think>Plain final answer."
        )

        self.assertTrue(parsed.is_well_formed)
        self.assertEqual(parsed.format_name, DEEPSEEK_THINK_COMPLETION_FORMAT)
        self.assertEqual(parsed.reasoning, "Evidence-window reasoning.")
        self.assertEqual(parsed.answer, "Plain final answer.")

    def test_everything_after_first_closing_boundary_is_answer(self):
        parsed = parse_structured_response(
            "First reasoning.</think>First answer line.\n"
            "Second line contains </think> as literal trailing text."
        )

        self.assertTrue(parsed.is_well_formed)
        self.assertEqual(parsed.reasoning, "First reasoning.")
        self.assertEqual(
            parsed.answer,
            "First answer line.\nSecond line contains </think> as literal trailing text.",
        )

    def test_closing_boundary_without_trailing_text_is_still_empty_answer(self):
        parsed = parse_structured_response("<think>Reasoning only.</think>   ")

        self.assertFalse(parsed.is_well_formed)
        self.assertEqual(parsed.reasoning, "Reasoning only.")
        self.assertEqual(parsed.answer, "")


class TestLengthTolerantCandidateExtraction(unittest.TestCase):
    def test_extracts_after_answer_without_requiring_think_close(self):
        result = extract_length_tolerant_candidate(
            "<think>unfinished reasoning\n<answer>Final policy text</answer>"
        )

        self.assertEqual(result.text, "Final policy text")
        self.assertEqual(result.extraction_mode, LENGTH_TOLERANT_ANSWER_TAG_MODE)
        self.assertTrue(result.has_leading_think_tag)
        self.assertTrue(result.has_answer_opening_tag)
        self.assertTrue(result.has_trailing_answer_closing_tag)
        self.assertTrue(result.is_nonempty)

    def test_answer_tags_are_case_insensitive_and_first_opening_wins(self):
        result = extract_length_tolerant_candidate(
            "<ThInK>reasoning<AnSwEr>first <answer>second</ANSWER>"
        )

        self.assertEqual(result.text, "first <answer>second")
        self.assertEqual(result.extraction_mode, LENGTH_TOLERANT_ANSWER_TAG_MODE)
        self.assertTrue(result.has_leading_think_tag)
        self.assertTrue(result.has_trailing_answer_closing_tag)

    def test_keeps_unclosed_answer_content(self):
        result = extract_length_tolerant_candidate(
            "<think>reasoning<answer>answer truncated at token limit"
        )

        self.assertEqual(result.text, "answer truncated at token limit")
        self.assertEqual(result.extraction_mode, LENGTH_TOLERANT_ANSWER_TAG_MODE)
        self.assertFalse(result.has_trailing_answer_closing_tag)
        self.assertTrue(result.is_nonempty)

    def test_falls_back_to_complete_nonempty_completion_without_answer_tag(self):
        completion = "<think>reasoning that reached the token limit"
        result = extract_length_tolerant_candidate(completion)

        self.assertEqual(result.text, completion)
        self.assertEqual(
            result.extraction_mode,
            LENGTH_TOLERANT_FULL_COMPLETION_MODE,
        )
        self.assertTrue(result.has_leading_think_tag)
        self.assertFalse(result.has_answer_opening_tag)
        self.assertFalse(result.has_trailing_answer_closing_tag)
        self.assertTrue(result.is_nonempty)

    def test_empty_and_non_string_inputs_are_empty(self):
        for value in ("", " \n\t ", None):
            with self.subTest(value=value):
                result = extract_length_tolerant_candidate(value)  # type: ignore[arg-type]
                self.assertEqual(result.text, "")
                self.assertEqual(result.extraction_mode, LENGTH_TOLERANT_EMPTY_MODE)
                self.assertFalse(result.is_nonempty)

    def test_empty_answer_does_not_fall_back_to_reasoning(self):
        result = extract_length_tolerant_candidate("<think>x<answer> </answer>")

        self.assertEqual(result.text, "")
        self.assertEqual(result.extraction_mode, LENGTH_TOLERANT_ANSWER_TAG_MODE)
        self.assertFalse(result.is_nonempty)

    def test_existing_strict_parser_still_rejects_missing_think_close(self):
        completion = "<think>unfinished reasoning<answer>candidate</answer>"

        strict = parse_structured_response(completion)
        self.assertFalse(strict.is_well_formed)
        self.assertEqual(extract_answer(completion), "")


if __name__ == "__main__":
    unittest.main()
