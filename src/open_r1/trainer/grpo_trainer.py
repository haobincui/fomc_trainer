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
import os
from collections.abc import Callable
from typing import List

from open_r1.configs import LoraArguments, GRPOConfig, GRPOScriptArguments
from open_r1.trainer.rewards.reward_register import get_reward_funcs
from open_r1.trainer.trainer import Trainer
from open_r1.utils.trl_compat import import_trl_grpo_symbols


from open_r1.utils.plot_loss import plot_reward_curve

GRPOTrainer, ModelConfig = import_trl_grpo_symbols()


class GrpoTrainer(Trainer):
    def __init__(self, script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: ModelConfig, peft_args: LoraArguments):
        super().__init__(script_args, training_args, model_args, peft_args)
        self.script_args = script_args
        self.training_args = training_args
        self.model_args = model_args
        self.peft_args = peft_args


        self._reward_funcs = None


    @property
    def reward_funcs(self):
        if self._reward_funcs is None:
            self._reward_funcs = self.load_reward_funcs()
        return self._reward_funcs


    def load_reward_funcs(self) -> List[Callable]:
        self.logger.info("*** Loading reward functions ***")
        reward_funcs = get_reward_funcs(self.script_args, self.training_args)
        return reward_funcs


    def load_trainer(self):
        import time
        self.logger.info("*** 🚀 Loading trainer ***")
        s = time.time()

        trainer = GRPOTrainer(
            model=self.model, # type: ignore
            reward_funcs=self.reward_funcs, # type: ignore
            args=self.training_args,
            train_dataset=self.dataset[self.script_args.dataset_train_split],
            eval_dataset=(
                self.dataset[self.script_args.dataset_test_split]
                if self.training_args.eval_strategy != "no"
                else None
            ),
            peft_config=self.peft_config,
            callbacks=self.load_callbacks(),
            processing_class=self.tokenizer,
        )
        e = time.time()
        elapsed = e - s
        self.logger.info(f"*** ✅ Loaded trainer, Time Usage {elapsed:.2f} s ***")
        return trainer

    def plot_customized_curve(self):
        reward_jsonl = os.path.join(self.training_args.output_dir, "reward_history.jsonl")
        save_plot = os.path.join(self.training_args.output_dir, "reward_curve.png")
        if os.path.exists(reward_jsonl):
            self.logger.info("📈 Plotting reward curve...")
            plot_reward_curve(reward_jsonl, save_plot)
            self.logger.info(f"✅ Reward curve saved to {save_plot}")
