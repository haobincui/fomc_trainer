import logging
import os
from typing import List
from vllm import LLM, SamplingParams



class Model:
    def __init__(self, model_path: str, temperature: float, top_p: float, max_new_tokens: int):
        self.model_path = model_path
        self.model = self._load_model()
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )

    def _load_model(self) -> LLM:
        return LLM(
            model=self.model_path,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )

    def generate_completion(self,prompt: List[str]) -> str:
        outputs = self.model.generate(prompt, self.sampling_params)
        if not outputs:
            raise ValueError("No outputs from the model.")
        return outputs[0].outputs[0].text


    def batch_generate_completion(self,batch_prompts: List[str]) -> List[str]:
        outputs = self.model.generate(batch_prompts, self.sampling_params)
        if not outputs:
            raise ValueError("No outputs from the model.")
        responses = []
        for output in outputs:
            responses.append(output.outputs[0].text)

        return responses

    def chat_completion(self, messages: List[dict]) -> str:
        outputs = self.model.chat([messages], self.sampling_params)
        if not outputs:
            raise ValueError("No outputs from the model.")
        response = outputs[0].outputs[0].text
        return response

    def batch_chat_completion(self, batch_messages: List[List[dict]]) -> List[str]:
        try:
            outputs = self.model.chat(batch_messages, self.sampling_params)
        except Exception as e:
            logging.warning(f"❌ vLLM chat batch failed: {e}")
            return ["Failed"] * len(batch_messages)

        responses = []
        if not outputs or len(outputs) != len(batch_messages):
            logging.warning(f"⚠️ Output length mismatch: expected {len(batch_messages)}, got {len(outputs)}")
            return ["Failed"] * len(batch_messages)

        for i, output in enumerate(outputs):
            try:
                if not output.outputs or not output.outputs[0].text:
                    responses.append("Failed")
                else:
                    responses.append(output.outputs[0].text)
            except Exception as e:
                logging.warning(f"❌ Failed to parse output at index {i}: {e}")
                responses.append("Failed")

        return responses






