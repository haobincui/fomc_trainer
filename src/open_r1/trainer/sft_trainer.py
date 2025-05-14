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

from peft import LoraConfig
from trl import SFTTrainer, ModelConfig

from open_r1.configs import LoraArguments, SFTConfig, SFTScriptArguments
from open_r1.trainer.trainer import Trainer
from open_r1.utils.callbacks import get_callbacks


class SftTrainer(Trainer):
    def __init__(self, script_args: SFTScriptArguments, training_args: SFTConfig, model_args: ModelConfig, peft_args: LoraArguments):
        super().__init__(script_args, training_args, model_args, peft_args)
        self.script_args = script_args
        self.training_args = training_args
        self.model_args = model_args
        self.peft_args = peft_args
        self.trainer = None



    def load_trainer(self):
        import time
        self.logger.info("*** 🚀 Loading trainer ***")
        s = time.time()

        def convert_chat(example):
            if isinstance(example["prompt"], list):
                return {"prompt": self._tokenizer.apply_chat_template(example["prompt"], tokenize=False)}
            return example

        self._dataset = self._dataset.map(convert_chat)
        self._dataset = self._dataset.rename_column("response", "completion")
        peft_config = LoraConfig(
                            r=self.peft_args.peft_r,
                            lora_alpha=self.peft_args.peft_lora_alpha,
                            lora_dropout=self.peft_args.peft_lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM",
                            target_modules=self.peft_args.peft_target_modules
                        )

        trainer = SFTTrainer(
            model=self._model,
            args=self.training_args,
            train_dataset=self._dataset[self.script_args.dataset_train_split],
            eval_dataset=(
                self._dataset[self.script_args.dataset_test_split]
                if self.training_args.eval_strategy != "no"
                else None
            ),
            processing_class=self._tokenizer,
            peft_config=peft_config,
            callbacks=get_callbacks(self.training_args, self.model_args), 
        )
        self.trainer = trainer
        e = time.time()
        self.logger.info(f"*** ✅ Loaded trainer, Time Usage {e - s} s ***")
        return trainer


