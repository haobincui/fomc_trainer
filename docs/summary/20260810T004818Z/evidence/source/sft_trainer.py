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


from trl import SFTTrainer

from open_r1.configs import LoraArguments, ModelConfig, SFTConfig, SFTScriptArguments
from open_r1.trainer.sft_prompt_renderer import render_sft_prompt
from open_r1.trainer.trainer import Trainer


class SftTrainer(Trainer):
    def __init__(self, script_args: SFTScriptArguments, training_args: SFTConfig, model_args: ModelConfig, peft_args: LoraArguments):
        super().__init__(script_args, training_args, model_args, peft_args)
        self.script_args = script_args
        self.training_args = training_args
        self.model_args = model_args
        self.peft_args = peft_args



    def load_trainer(self):
        import time
        self.logger.info("*** 🚀 Loading trainer ***")
        s = time.time()

        def convert_chat(example):
            if isinstance(example["prompt"], list):
                return {
                    "prompt": render_sft_prompt(self.tokenizer, example["prompt"])
                }
            return example

        dataset = self.dataset.map(convert_chat).rename_column("response", "completion")

        trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset[self.script_args.dataset_train_split],
            eval_dataset=(
                dataset[self.script_args.dataset_test_split]
                if self.training_args.eval_strategy != "no"
                else None
            ),
            processing_class=self.tokenizer,
            peft_config=self.peft_config,
            callbacks=self.load_callbacks(),
        )
        e = time.time()
        elapsed = e - s
        self.logger.info(f"*** ✅ Loaded trainer, Time Usage {elapsed:.2f} s ***")
        return trainer
    
    def plot_customized_curve(self):
        pass
