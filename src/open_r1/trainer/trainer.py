import logging
import os
import sys
from abc import ABC, abstractmethod

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.data_loader import load_train_eval_datasets
from open_r1.utils import get_model, get_tokenizer
from open_r1.utils.wandb_logging import init_wandb_training




class Trainer(ABC):


    def __init__(self, script_args, training_args, model_args, peft_args):
        self.script_args = script_args
        self.training_args = training_args
        self.model_args = model_args
        self.peft_args = peft_args
        self.logger = self._set_logger()

        self.trainer = None

        self._model = self.load_model()
        self._tokenizer = self.load_tokenizer()
        self._dataset = self.load_dataset()


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

        self.logger = logger
        return logger


    def load_checkpoint(self):
        last_checkpoint = None
        if os.path.isdir(self.training_args.output_dir):
            last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
        if last_checkpoint is not None and self.training_args.resume_from_checkpoint is None:
            self.logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

        if "wandb" in self.training_args.report_to:
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
        dataset = dataset.rename_column("response", "completion")


        for split in dataset:
            if "messages" in dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns("messages")

        return dataset

    def load_tokenizer(self):
        self.logger.info("*** Loading tokenizer ***")
        tokenizer = get_tokenizer(self.model_args, self.training_args)
        return tokenizer

    def load_model(self):
        self.logger.info("*** Loading model ***")
        model = get_model(self.model_args, self.training_args)
        return model

    @abstractmethod
    def load_trainer(self):
        pass


    def start_train(self):
        last_checkpoint = self.load_checkpoint()
        self.logger.info("*** 🚀 Start Training ***")
        checkpoint = None
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(self._dataset[self.script_args.dataset_train_split])
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
            metrics["eval_samples"] = len(self._dataset[self.script_args.dataset_test_split])
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








