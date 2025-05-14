# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Callable
from typing import List

from open_r1.trainer.rewards.reward_register import get_reward_funcs
from open_r1.trainer.trainer import Trainer
from open_r1.utils.callbacks import get_callbacks
from open_r1.configs import LoraArguments, GRPOConfig, GRPOScriptArguments

from trl import GRPOTrainer, get_peft_config, ModelConfig
from peft import LoraConfig







class GrpoTrainer(Trainer):
    def __init__(self, script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: ModelConfig, peft_args: LoraArguments):
        super().__init__(script_args, training_args, model_args, peft_args)
        self.script_args = script_args
        self.training_args = training_args
        self.model_args = model_args
        self.peft_args = peft_args
        self.trainer = None

        self._reward_funcs = self.load_reward_funcs()


    def load_reward_funcs(self) -> List[Callable]:
        self.logger.info("*** Loading reward functions ***")
        reward_funcs = get_reward_funcs(self.script_args)
        return reward_funcs


    def load_trainer(self):
        import time
        self.logger.info("*** 🚀 Loading trainer ***")
        s = time.time()

        peft_config = LoraConfig(
                            r=self.peft_args.peft_r,
                            lora_alpha=self.peft_args.peft_lora_alpha,
                            lora_dropout=self.peft_args.peft_lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM",
                            target_modules=self.peft_args.peft_target_modules
                        )
        reward_kwargs = {}
        if self.script_args.save_reward:
            reward_kwargs["save_path"] = f"{self.training_args.output_dir}/reward.jsonl"


        trainer = GRPOTrainer(
            model=self._model,
            reward_funcs=self._reward_funcs,
            reward_kwargs=reward_kwargs,
            args=self.training_args,
            train_dataset=self._dataset[self.script_args.dataset_train_split],
            eval_dataset=(
                self._dataset[self.script_args.dataset_test_split]
                if self.training_args.eval_strategy != "no"
                else None
            ),
            peft_config=peft_config,
            callbacks=get_callbacks(self.training_args, self.model_args),
            processing_class=self._tokenizer,
        )
        self.trainer = trainer
        e = time.time()
        self.logger.info(f"*** ✅ Loaded trainer, Time Usage {e - s} s ***")
        return trainer


