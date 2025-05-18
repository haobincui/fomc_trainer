import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Union

import datasets
import transformers
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig
from trl import ScriptArguments
from peft import LoraConfig, PeftModel

from open_r1.configs import GRPOConfig, LoraArguments, SFTConfig
from open_r1.data_loader import load_train_eval_datasets
from open_r1.utils import get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.plot_loss import plot_training_curve
from open_r1.utils.wandb_logging import init_wandb_training


class Trainer(ABC):


    def __init__(self, script_args: ScriptArguments, training_args: Union[SFTConfig, GRPOConfig], model_args: ModelConfig, peft_args: LoraArguments):
        self.script_args = script_args
        self.training_args = training_args
        self.model_args = model_args
        self.peft_args = peft_args


        self._trainer = None

        self._logger = None
        self._model = None
        self._tokenizer = None
        self._dataset = None
        self._peft_config = None

    @property
    def logger(self):
        if self._logger is None:
            self._logger = self._set_logger()
        return self._logger

    @property
    def model(self):
        if self._model is None:
            self._model = self.load_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.load_tokenizer()
        return self._tokenizer

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = self.load_dataset()
        return self._dataset

    @property
    def trainer(self):
        if self._trainer is None:
            self._trainer = self.load_trainer()
        return self._trainer


    @property
    def peft_config(self):
        if self._peft_config is None:
            self._peft_config = self.load_peft_config()
        return self._peft_config

    def _set_logger(self):
        # create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(self.training_args.get_process_log_level())

        # add stdout handler
        if not logger.handlers:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(logging.Formatter(
                fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            logger.addHandler(stream_handler)

        # set datasets + transformers logging
        log_level = self.training_args.get_process_log_level()
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # print
        logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}"
            + f" distributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {self.training_args.fp16}"
        )
        logger.info(f"Model parameters {self.model_args}")
        logger.info(f"Script parameters {self.script_args}")
        logger.info(f"Training parameters {self.training_args}")
        logger.info(f"Peft parameters {self.peft_args}")

        self._logger = logger
        return logger
    


    def load_checkpoint(self, checkpoint_path: str = None):
        last_checkpoint = None
        if os.path.isdir(checkpoint_path): # type: ignore
            last_checkpoint = get_last_checkpoint(checkpoint_path)
        if last_checkpoint is not None and self.training_args.resume_from_checkpoint is None:
            self.logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

        if "wandb" in self.training_args.report_to: # type: ignore
            init_wandb_training(self.training_args)
        return last_checkpoint


    def load_dataset(self):
        self.logger.info("*** Load dataset ***")
        dataset = load_train_eval_datasets(self.script_args.dataset_name)

        def _make_conversation(
                example, prompt_column: str = self.script_args.dataset_prompt_column
        ):
            prompt = []

            if self.training_args.system_prompt is not None:
                prompt.append({"role": "system", "content": self.training_args.system_prompt})

            if prompt_column not in example:
                raise ValueError(
            f"Dataset column '{prompt_column}' not found. Available columns: {list(example.keys())}"
        )

            prompt.append({"role": "user", "content": example[prompt_column]})
            return {"prompt": prompt}

        dataset = dataset.map(_make_conversation)


        for split in dataset:
            if "messages" in dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns("messages")

        return dataset

    def load_tokenizer(self):
        self.logger.info("*** Loading tokenizer ***")
        tokenizer = get_tokenizer(self.model_args, self.training_args)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self):
        self.logger.info("*** Loading model ***")
        model = get_model(self.model_args, self.training_args)
        return model

    def load_peft_config(self):
        self.logger.info(f"🛠️ Initializing new LoRA config")
        peft_config = LoraConfig(
                            r=self.peft_args.peft_r,
                            lora_alpha=self.peft_args.peft_lora_alpha,
                            lora_dropout=self.peft_args.peft_lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM",
                            target_modules=self.peft_args.peft_target_modules
                        )
        return peft_config
    
    def patch_model_and_tokenizer(self):
        from peft import prepare_model_for_kbit_training

        self.logger.info("*** Patching model for 4-bit + gradient checkpointing ***")

        # ✅ 必须在 tokenizer 设置后同步 pad_token_id 到 model.config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # ✅ 使用 PEFT 的官方函数做 checkpointing 安全封装
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.training_args.gradient_checkpointing
        )



    @abstractmethod
    def load_trainer(self):
        raise NotImplementedError("load_trainer() must be implemented in subclasses")
    
    @abstractmethod
    def plot_customized_curve(self):
        pass


    def load_callbacks(self):
        try:
            callbacks = get_callbacks(self.training_args, self.model_args)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load callbacks, continuing without: {e}")
            callbacks = None
        return callbacks


    def start_train(self):
        last_checkpoint = self.load_checkpoint(self.training_args.output_dir)

        self.logger.info("*** 🚀 Start Training ***")
        checkpoint = None
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(self.dataset[self.script_args.dataset_train_split])
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
        

        self.logger.info("*** 🚀 Save model ***")
        self.trainer.save_model(self.training_args.output_dir)
        self.logger.info(f"✅ Model saved to {self.training_args.output_dir}")

        # Save everything else on main process
        kwargs = {
            "dataset_name": self.script_args.dataset_name,
            "tags": ["open-r1"],
        }
        if self.trainer.accelerator.is_main_process:
            self.trainer.create_model_card(**kwargs)
            # Restore k,v cache for fast inference
            self.trainer.model.config.use_cache = True
            self.trainer.model.config.save_pretrained(self.training_args.output_dir)

        ##########
        # Evaluate
        ##########
        if self.training_args.do_eval:
            self.logger.info("*** 🚀 Evaluating ***")
            metrics = self.trainer.evaluate()
            metrics["eval_samples"] = len(self.dataset[self.script_args.dataset_test_split])
            self.trainer.log_metrics("eval", metrics)
            self.trainer.save_metrics("eval", metrics)
            self.logger.info("*** ✅ Evaluated ***")

        #############
        # push to hub
        #############
        if self.training_args.push_to_hub:
            self.logger.info("Pushing to hub...")
            self.trainer.push_to_hub(**kwargs)
        self.logger.info("✅ Training completed successfully.")



        #############
        # plot loss curve
        #############

        if self.trainer.accelerator.is_main_process:

            loss_jsonl = os.path.join(self.training_args.output_dir, "loss_history.jsonl")
            save_plot = os.path.join(self.training_args.output_dir, "training_curve.png")
            if os.path.exists(loss_jsonl):
                self.logger.info("📈 Plotting training curve...")
                plot_training_curve(loss_jsonl, save_plot)
                self.logger.info(f"✅ Training curve saved to {save_plot}")
            self.plot_customized_curve()



    def export_model(self):
        if not self.peft_args.peft_merged_model_path:
            self.logger.warning("❌ Merged model path is not set. Skipping merge.")
            return

        model = self.trainer.model.merge_and_unload()
        model.save_pretrained(self.peft_args.peft_merged_model_path)

        self.logger.info(f"✅ Merged model saved to {self.peft_args.peft_merged_model_path}")












