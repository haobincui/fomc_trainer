import unittest
from unittest.mock import patch

from generate_new_response import generate_new_response


class TestGenerateNewResponse(unittest.TestCase):
    @patch("generate_new_response.generate_responses")
    def test_failed_batch_outputs_are_not_counted_as_success(self, mock_generate_responses):
        mock_generate_responses.return_value = ["good output", "Failed", ""]
        input_rows = [
            {"prompt": "p1", "response": "r1"},
            {"prompt": "p2", "response": "r2"},
            {"prompt": "p3", "response": "r3"},
        ]

        results = generate_new_response(input_rows, "models/test-model", batch_size=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["generated"], "good output")
        self.assertNotIn("generated", input_rows[0])
        self.assertNotIn("generated", input_rows[1])

    @patch("generate_new_response.generate_responses")
    def test_legacy_argument_order_is_normalized(self, mock_generate_responses):
        mock_generate_responses.return_value = ["generated text"]
        input_rows = [{"prompt": "prompt", "response": "target"}]

        results = generate_new_response(input_rows, "output/test.jsonl", "models/test-model", batch_size=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["generated"], "generated text")


if __name__ == "__main__":
    unittest.main()
