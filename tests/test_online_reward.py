import os
import unittest
from unittest.mock import Mock, patch

from open_r1.trainer.rewards.reward_funcs.online_reward import (
    _parse_answer,
    _parse_reasoning_and_answer,
    _parse_score,
    _send_eval_request,
    get_online_reward_settings,
)


class TestOnlineRewardHelpers(unittest.TestCase):
    def test_parse_reasoning_and_answer_from_tagged_output(self):
        reasoning, answer = _parse_reasoning_and_answer(
            "<think>step one</think><answer>final answer</answer>"
        )
        self.assertEqual(reasoning, "step one")
        self.assertEqual(answer, "final answer")

    def test_parse_reasoning_and_answer_from_gemini_output(self):
        reasoning, answer = _parse_reasoning_and_answer(
            "<|channel>thought\nstep one\n<channel|>final answer"
        )
        self.assertEqual(reasoning, "step one")
        self.assertEqual(answer, "final answer")

    def test_parse_reasoning_and_answer_from_deepseek_completion_suffix(self):
        reasoning, answer = _parse_reasoning_and_answer(
            "step one\nstep two\n</think>\nfinal answer"
        )
        self.assertEqual(reasoning, "step one\nstep two")
        self.assertEqual(answer, "final answer")

    def test_parse_answer_returns_answer_segment_only(self):
        answer = _parse_answer("<think>x</think><answer>policy text</answer>")
        self.assertEqual(answer, "policy text")

    def test_parse_answer_returns_gemini_answer_segment_only(self):
        answer = _parse_answer("<|channel>thought\nx\n<channel|>policy text")
        self.assertEqual(answer, "policy text")

    def test_parse_score_reads_boxed_integer(self):
        self.assertEqual(_parse_score("**Total Score**: \\boxed{32}"), 32.0)

    def test_online_reward_settings_follow_environment_fallbacks(self):
        with patch.dict(
            os.environ,
            {
                "OPEN_R1_JUDGE_URL": "http://localhost:9999/api/chat/",
                "OPEN_R1_JUDGE_MODEL": "judge-model",
                "OPEN_R1_JUDGE_TIMEOUT": "12",
                "OPEN_R1_JUDGE_SLEEP_SECONDS": "0.25",
                "OPEN_R1_JUDGE_VERBOSE": "1",
                "OPEN_R1_JUDGE_API_KEY": "secret",
            },
            clear=False,
        ):
            settings = get_online_reward_settings()

        self.assertEqual(settings["url"], "http://localhost:9999/api/chat/")
        self.assertEqual(settings["model"], "judge-model")
        self.assertEqual(settings["timeout"], 12)
        self.assertEqual(settings["sleep_seconds"], 0.25)
        self.assertTrue(settings["verbose"])
        self.assertEqual(settings["api_key"], "secret")

    @patch("open_r1.trainer.rewards.reward_funcs.online_reward.requests.post")
    def test_send_eval_request_handles_ollama_style_payload(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"message": {"content": "**Total Score**: \\boxed{35}"}}
        mock_post.return_value = mock_response

        score = _send_eval_request(
            "prompt",
            url="http://localhost:11432/api/chat/",
            model="gemma3:12b",
            timeout=30,
            api_key=None,
            verbose=False,
        )

        self.assertEqual(score, 35.0)
        _, kwargs = mock_post.call_args
        self.assertIn("keep_alive", kwargs["json"])

    @patch("open_r1.trainer.rewards.reward_funcs.online_reward.requests.post")
    def test_send_eval_request_handles_openai_style_payload(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "**Total Score**: \\boxed{28}"}}]
        }
        mock_post.return_value = mock_response

        score = _send_eval_request(
            "prompt",
            url="http://127.0.0.1:8000/v1/chat/completions",
            model="models/gemma-3-12b-it",
            timeout=30,
            api_key=None,
            verbose=False,
        )

        self.assertEqual(score, 28.0)
        _, kwargs = mock_post.call_args
        self.assertNotIn("keep_alive", kwargs["json"])


if __name__ == "__main__":
    unittest.main()
