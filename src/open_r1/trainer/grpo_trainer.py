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
from trl import GRPOTrainer, get_peft_config






class GrpoTrainer(Trainer):
    def __init__(self, script_args, training_args, model_args):
        super().__init__(script_args, training_args, model_args)
        self.script_args = script_args
        self.training_args = training_args
        self.model_args = model_args
        self.trainer = None

        self._reward_funcs = self.load_reward_funcs()


    def load_reward_funcs(self) -> List[Callable]:
        self.logger.info("*** Loading reward functions ***")
        reward_funcs = get_reward_funcs(self.script_args)
        return reward_funcs


    def load_trainer(self):

        self.logger.info("*** Loading trainer ***")
        trainer = GRPOTrainer(
            model=self._model,
            reward_funcs=self._reward_funcs,
            args=self.training_args,
            train_dataset=self._dataset[self.script_args.dataset_train_split],
            eval_dataset=(
                self._dataset[self.script_args.dataset_test_split]
                if self.training_args.eval_strategy != "no"
                else None
            ),
            peft_config=get_peft_config(self.model_args),
            callbacks=get_callbacks(self.training_args, self.model_args),
            processing_class=self._tokenizer,
        )
        self.trainer = trainer
        return trainer


