import logging
from typing import Any, List
from vllm import LLM, SamplingParams



class Model:
    def __init__(
        self,
        model_path: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        max_model_len: int = 16384,
        tokenizer_path: str | None = None,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_model_len = int(max_model_len)
        self.tokenizer_path = tokenizer_path
        self.model = self._load_model()
        self.sampling_params = self._sampling_params()

    def _sampling_params(self, seed: int | None = None) -> SamplingParams:
        parameters = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
        }
        if seed is not None:
            parameters["seed"] = int(seed)
        return SamplingParams(
            **parameters,
        )

    def _load_model(self) -> LLM:
        return LLM(
            model=self.model_path,
            tokenizer=self.tokenizer_path or self.model_path,
            dtype="bfloat16",
            max_model_len=self.max_model_len,
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
        )

    @staticmethod
    def _completion_metadata(
        output: Any,
        *,
        expected_prompt_token_count: int | None = None,
    ) -> dict[str, Any]:
        completion = output.outputs[0]
        prompt_token_ids = getattr(output, "prompt_token_ids", None)
        completion_token_ids = getattr(completion, "token_ids", None)
        observed_prompt_token_count = (
            len(prompt_token_ids) if prompt_token_ids is not None else None
        )
        if (
            expected_prompt_token_count is not None
            and observed_prompt_token_count != expected_prompt_token_count
        ):
            raise ValueError(
                "vLLM consumed a different prompt-token count than the "
                "preflight chat template: "
                f"expected={expected_prompt_token_count}, "
                f"observed={observed_prompt_token_count}"
            )
        return {
            "text": completion.text,
            "finish_reason": getattr(completion, "finish_reason", None),
            "stop_reason": getattr(completion, "stop_reason", None),
            "prompt_token_count": observed_prompt_token_count,
            "prompt_preflight_token_count": expected_prompt_token_count,
            "output_token_count": (
                len(completion_token_ids)
                if completion_token_ids is not None
                else None
            ),
            "input_was_truncated": False,
        }

    @staticmethod
    def _failed_metadata(reason: str = "generation_error") -> dict[str, Any]:
        return {
            "text": "Failed",
            "finish_reason": reason,
            "stop_reason": None,
            "prompt_token_count": None,
            "prompt_preflight_token_count": None,
            "output_token_count": None,
            "input_was_truncated": None,
        }

    def _preflight_chat_prompt(self, messages: List[dict]) -> int:
        tokenizer = self.model.get_tokenizer()
        prompt_token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        token_count = len(prompt_token_ids)
        requested_total = token_count + self.max_new_tokens
        if requested_total > self.max_model_len:
            raise ValueError(
                "Generation request exceeds the frozen context budget before "
                f"inference: prompt_tokens={token_count}, "
                f"max_new_tokens={self.max_new_tokens}, "
                f"max_model_len={self.max_model_len}, "
                f"overflow={requested_total - self.max_model_len}"
            )
        return token_count

    def generate_completion(self, prompt: List[str], seed: int | None = None) -> str:
        outputs = self.model.generate(prompt, self._sampling_params(seed))
        if not outputs:
            raise ValueError("No outputs from the model.")
        return outputs[0].outputs[0].text


    def batch_generate_completion(
        self,
        batch_prompts: List[str],
        seed: int | None = None,
    ) -> List[str]:
        outputs = self.model.generate(batch_prompts, self._sampling_params(seed))
        if not outputs:
            raise ValueError("No outputs from the model.")
        responses = []
        for output in outputs:
            responses.append(output.outputs[0].text)

        return responses

    def chat_completion(self, messages: List[dict], seed: int | None = None) -> str:
        outputs = self.model.chat([messages], self._sampling_params(seed))
        if not outputs:
            raise ValueError("No outputs from the model.")
        response = outputs[0].outputs[0].text
        return response

    def batch_chat_completion(
        self,
        batch_messages: List[List[dict]],
        seed: int | None = None,
        *,
        seeds: List[int] | None = None,
        return_metadata: bool = False,
    ) -> List[str] | List[dict[str, Any]]:
        if seed is not None and seeds is not None:
            raise ValueError("Pass either seed or per-row seeds, not both")
        if seeds is not None and len(seeds) != len(batch_messages):
            raise ValueError(
                "Per-row seed count must match the number of batch messages"
            )
        preflight_prompt_token_counts = [
            self._preflight_chat_prompt(messages)
            for messages in batch_messages
        ]
        sampling_params: SamplingParams | List[SamplingParams]
        if seeds is None:
            sampling_params = self._sampling_params(seed)
        else:
            sampling_params = [self._sampling_params(value) for value in seeds]
        try:
            outputs = self.model.chat(batch_messages, sampling_params)
        except Exception as e:
            logging.warning(f"❌ vLLM chat batch failed: {e}")
            failures = [
                self._failed_metadata() for _ in range(len(batch_messages))
            ]
            return failures if return_metadata else ["Failed"] * len(batch_messages)

        responses: list[str] | list[dict[str, Any]] = []
        if not outputs or len(outputs) != len(batch_messages):
            logging.warning(f"⚠️ Output length mismatch: expected {len(batch_messages)}, got {len(outputs)}")
            failures = [
                self._failed_metadata("output_count_mismatch")
                for _ in range(len(batch_messages))
            ]
            return failures if return_metadata else ["Failed"] * len(batch_messages)

        for i, output in enumerate(outputs):
            try:
                if not output.outputs or not output.outputs[0].text:
                    response = self._failed_metadata("empty_output")
                else:
                    response = self._completion_metadata(
                        output,
                        expected_prompt_token_count=(
                            preflight_prompt_token_counts[i]
                        ),
                    )
            except Exception as e:
                logging.warning(f"❌ Failed to parse output at index {i}: {e}")
                response = self._failed_metadata("parse_error")
            responses.append(response if return_metadata else response["text"])

        return responses
