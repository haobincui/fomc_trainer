import hashlib
import json
import logging
import os
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Union

import datasets
import transformers
import torch
from transformers.trainer_utils import get_last_checkpoint
from trl import get_quantization_config
from trl import ScriptArguments
from peft import LoraConfig

from open_r1.configs import GRPOConfig, LoraArguments, ModelConfig, SFTConfig
from open_r1.data_loader import load_train_eval_datasets
from open_r1.trainer.dataset_release import (
    CHK4_STUDENT_SYSTEM_PROMPT_SHA256,
    STANDALONE_CHK3_BINDING_SCHEMA,
    STANDALONE_CHK3_DIRECT_SCOPE,
    verify_chk1_semantic_override,
    verify_chk4_release_for_role,
    verify_clean_sft_release,
    verify_standalone_chk3_direct_sft_release,
)
from open_r1.trainer.fixed_schedule_sampler import (
    SUPPORTED_SAMPLER_TYPES as FIXED_SCHEDULE_SAMPLER_TYPES,
    validate_runtime_contract as validate_fixed_schedule_runtime,
)
from open_r1.trainer.fixed_schedule_sampler_v3 import (
    SAMPLER_TYPE as CORRECTION_FIXED_SCHEDULE_SAMPLER_TYPE,
    validate_runtime_contract as validate_correction_fixed_schedule_runtime,
)
from open_r1.trainer.rewards.reward_funcs.online_reward import (
    get_online_reward_settings,
)
from open_r1.trainer.prompt_contract import compose_user_prompt
from open_r1.utils import get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.plot_loss import plot_training_curve
from open_r1.utils.wandb_logging import init_wandb_training


class Trainer(ABC):
    _SEMANTIC_OVERRIDE_FIELDS = (
        "dataset_semantic_override_stage",
        "dataset_semantic_override_candidate_manifest",
        "dataset_semantic_override_candidate_manifest_sha256",
        "dataset_semantic_override_validation_receipt",
        "dataset_semantic_override_validation_receipt_sha256",
        "dataset_semantic_override_audit_summary",
        "dataset_semantic_override_audit_summary_sha256",
        "dataset_semantic_override_authorization_receipt",
        "dataset_semantic_override_authorization_receipt_sha256",
    )
    _STANDALONE_CHK3_FIELDS = (
        "dataset_standalone_chk3_scope",
        "dataset_standalone_chk3_release_manifest",
        "dataset_standalone_chk3_release_manifest_sha256",
    )
    _CHK4_DECISION_FIELDS = (
        "dataset_chk4_role",
        "dataset_chk4_release_manifest",
        "dataset_chk4_release_manifest_sha256",
    )

    def __init__(
        self,
        script_args: ScriptArguments,
        training_args: Union[SFTConfig, GRPOConfig],
        model_args: ModelConfig,
        peft_args: LoraArguments,
    ):
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
        self._chk4_release_binding = None

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
            stream_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
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
        if os.path.isdir(checkpoint_path):  # type: ignore
            last_checkpoint = get_last_checkpoint(checkpoint_path)
        if (
            last_checkpoint is not None
            and self.training_args.resume_from_checkpoint is None
        ):
            self.logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}."
            )

        if "wandb" in self.training_args.report_to:  # type: ignore
            init_wandb_training(self.training_args)
        return last_checkpoint

    def _dataset_binding(self) -> tuple[str, dict[str, object]]:
        release_manifest = getattr(self.script_args, "dataset_release_manifest", None)
        release_manifest_sha256 = getattr(
            self.script_args, "dataset_release_manifest_sha256", None
        )
        if bool(release_manifest) != bool(release_manifest_sha256):
            raise ValueError(
                "dataset_release_manifest and dataset_release_manifest_sha256 "
                "must be configured together"
            )
        standalone_chk3 = {
            field: getattr(self.script_args, field, None)
            for field in self._STANDALONE_CHK3_FIELDS
        }
        configured_standalone_fields = {
            field
            for field, value in standalone_chk3.items()
            if value is not None and value != ""
        }
        if configured_standalone_fields and len(configured_standalone_fields) != len(
            standalone_chk3
        ):
            missing = sorted(set(standalone_chk3) - configured_standalone_fields)
            raise ValueError(
                "standalone chk3 release fields must all be configured; missing: "
                + ", ".join(missing)
            )
        override = {
            field: getattr(self.script_args, field, None)
            for field in self._SEMANTIC_OVERRIDE_FIELDS
        }
        configured_override_fields = {
            field
            for field, value in override.items()
            if value is not None and value != ""
        }
        if configured_override_fields and len(configured_override_fields) != len(
            override
        ):
            missing = sorted(set(override) - configured_override_fields)
            raise ValueError(
                "chk1 semantic override fields must all be configured; missing: "
                + ", ".join(missing)
            )
        chk4_decision = {
            field: getattr(self.script_args, field, None)
            for field in self._CHK4_DECISION_FIELDS
        }
        configured_chk4_fields = {
            field
            for field, value in chk4_decision.items()
            if value is not None and value != ""
        }
        if configured_chk4_fields and len(configured_chk4_fields) != len(chk4_decision):
            missing = sorted(set(chk4_decision) - configured_chk4_fields)
            raise ValueError(
                "chk4 Decision release fields must all be configured; missing: "
                + ", ".join(missing)
            )
        configured_modes = sum(
            (
                bool(release_manifest),
                bool(configured_standalone_fields),
                bool(configured_override_fields),
                bool(configured_chk4_fields),
            )
        )
        if configured_modes > 1:
            raise ValueError(
                "clean release, standalone chk3 release, chk1 semantic override, "
                "and chk4 Decision bindings are mutually exclusive"
            )
        if configured_chk4_fields:
            if getattr(self.script_args, "dataset_prompt_column", None) != "prompt":
                raise ValueError(
                    "chk4 Decision release requires dataset_prompt_column=prompt"
                )
            if getattr(self.script_args, "dataset_train_split", None) != "train":
                raise ValueError(
                    "chk4 Decision release requires dataset_train_split=train"
                )
            if getattr(self.script_args, "dataset_test_split", None) != "validation":
                raise ValueError(
                    "chk4 Decision release requires dataset_test_split=validation"
                )
            if getattr(self.script_args, "user_prompt_suffix", None):
                raise ValueError(
                    "chk4 Decision release forbids user_prompt_suffix prompt drift"
                )
            return "chk4_decision_release", chk4_decision
        if configured_override_fields:
            return "chk1_semantic_override", override
        if configured_standalone_fields:
            return "standalone_chk3_direct_non_promotable", standalone_chk3
        if release_manifest:
            return "clean_release", {
                "manifest": release_manifest,
                "manifest_sha256": release_manifest_sha256,
            }
        return "legacy_unbound", {}

    def _verify_chk4_release_binding(
        self, binding: dict[str, object]
    ) -> dict[str, object]:
        cached = getattr(self, "_chk4_release_binding", None)
        if cached is None:
            cached = verify_chk4_release_for_role(
                dataset_dir=self.script_args.dataset_name,
                manifest_path=binding["dataset_chk4_release_manifest"],
                expected_manifest_sha256=binding[
                    "dataset_chk4_release_manifest_sha256"
                ],
                dataset_role=str(binding["dataset_chk4_role"]),
                system_prompt=self.training_args.system_prompt,
                model_path=self.model_args.model_name_or_path,
            )
            self._chk4_release_binding = cached
        return cached

    def load_dataset(self):
        self.logger.info("*** Load dataset ***")
        binding_mode, binding = self._dataset_binding()
        manifest_split_files = None
        if binding_mode == "clean_release":
            release = verify_clean_sft_release(
                dataset_dir=self.script_args.dataset_name,
                manifest_path=binding["manifest"],
                expected_manifest_sha256=binding["manifest_sha256"],
            )
            self.logger.info(
                "Verified immutable clean SFT release %s",
                release.get("release_id", "unknown"),
            )
            manifest_root = Path(str(binding["manifest"])).expanduser().resolve().parent
            split_records = release["split_files"]
            manifest_split_files = {
                "train": manifest_root / split_records["train"]["path"],
                "validation": manifest_root / split_records["eval"]["path"],
            }
        elif binding_mode == "chk1_semantic_override":
            release = verify_chk1_semantic_override(
                dataset_dir=self.script_args.dataset_name,
                candidate_manifest_path=binding[
                    "dataset_semantic_override_candidate_manifest"
                ],
                expected_candidate_manifest_sha256=binding[
                    "dataset_semantic_override_candidate_manifest_sha256"
                ],
                deterministic_validation_path=binding[
                    "dataset_semantic_override_validation_receipt"
                ],
                expected_deterministic_validation_sha256=binding[
                    "dataset_semantic_override_validation_receipt_sha256"
                ],
                semantic_audit_summary_path=binding[
                    "dataset_semantic_override_audit_summary"
                ],
                expected_semantic_audit_summary_sha256=binding[
                    "dataset_semantic_override_audit_summary_sha256"
                ],
                authorization_receipt_path=binding[
                    "dataset_semantic_override_authorization_receipt"
                ],
                expected_authorization_receipt_sha256=binding[
                    "dataset_semantic_override_authorization_receipt_sha256"
                ],
                training_stage=str(binding["dataset_semantic_override_stage"]),
            )
            self.logger.warning(
                "Verified explicit chk1-only semantic override for candidate %s; "
                "this dataset is prohibited for chk2/chk3/chk4",
                release.get("release_id", "unknown"),
            )
            candidate_root = (
                Path(str(binding["dataset_semantic_override_candidate_manifest"]))
                .expanduser()
                .resolve()
                .parent
            )
            split_records = release["split_files"]
            manifest_split_files = {
                "train": candidate_root / split_records["train"]["path"],
                "validation": candidate_root / split_records["eval"]["path"],
            }
        elif binding_mode == "standalone_chk3_direct_non_promotable":
            release = verify_standalone_chk3_direct_sft_release(
                dataset_dir=self.script_args.dataset_name,
                manifest_path=binding["dataset_standalone_chk3_release_manifest"],
                expected_manifest_sha256=binding[
                    "dataset_standalone_chk3_release_manifest_sha256"
                ],
                training_scope=str(binding["dataset_standalone_chk3_scope"]),
            )
            self.logger.warning(
                "Verified standalone chk1-to-chk3 direct-SFT release %s; "
                "this binding is non-canonical, non-DAG-bindable, and non-promotable",
                release.get("release_id", "unknown"),
            )
            manifest_root = (
                Path(str(binding["dataset_standalone_chk3_release_manifest"]))
                .expanduser()
                .resolve()
                .parent
            )
            split_records = release["split_files"]
            manifest_split_files = {
                "train": manifest_root / split_records["train"]["path"],
                "validation": manifest_root / split_records["validation"]["path"],
            }
        elif binding_mode == "chk4_decision_release":
            release = self._verify_chk4_release_binding(binding)
            self.logger.info(
                "Verified immutable chk4 Decision release %s for role %s; "
                "test remains sealed evaluation-only",
                release["release_id"],
                release["dataset_role"],
            )
            manifest_split_files = release["split_files"]
        dataset = load_train_eval_datasets(
            self.script_args.dataset_name,
            split_files=manifest_split_files,
        )

        def _make_conversation(
            example, prompt_column: str = self.script_args.dataset_prompt_column
        ):
            prompt = []

            if self.training_args.system_prompt is not None:
                prompt.append(
                    {"role": "system", "content": self.training_args.system_prompt}
                )

            if prompt_column not in example:
                raise ValueError(
                    f"Dataset column '{prompt_column}' not found. Available columns: {list(example.keys())}"
                )

            user_prompt = compose_user_prompt(
                example[prompt_column],
                getattr(self.script_args, "user_prompt_suffix", None),
            )
            prompt.append({"role": "user", "content": user_prompt})
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
        if get_quantization_config(self.model_args) is not None:
            model = self.patch_model_and_tokenizer(model, self.tokenizer)
        return model

    def load_peft_config(self):
        self.logger.info("🛠️ Initializing new LoRA config")
        if self.peft_args.peft_bias != "none":
            raise ValueError("retrain-v2 supports only peft_bias=none")
        peft_config = LoraConfig(
            r=self.peft_args.peft_r,
            lora_alpha=self.peft_args.peft_lora_alpha,
            lora_dropout=self.peft_args.peft_lora_dropout,
            bias=self.peft_args.peft_bias,
            task_type="CAUSAL_LM",
            target_modules=self.peft_args.peft_target_modules,
        )
        return peft_config

    def patch_model_and_tokenizer(self, model, tokenizer):
        from peft import prepare_model_for_kbit_training

        self.logger.info("*** Patching model for 4-bit + gradient checkpointing ***")

        model.config.pad_token_id = tokenizer.pad_token_id

        return prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=self.training_args.gradient_checkpointing
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
            if "checkpoint_retention" in getattr(self.training_args, "callbacks", []):
                self.logger.error(
                    "Checkpoint retention callback failed to load; aborting: %s",
                    e,
                )
                raise
            self.logger.warning(f"⚠️ Failed to load callbacks, continuing without: {e}")
            callbacks = None
        return callbacks

    def _runtime_config_path(self) -> str:
        return os.path.join(
            self.training_args.output_dir, "resolved_runtime_config.json"
        )

    def _serialize_args(self, args):
        if is_dataclass(args):
            return asdict(args)
        if hasattr(args, "__dict__"):
            return dict(vars(args))
        return str(args)

    def _build_runtime_config(self) -> dict:
        quantization_config = get_quantization_config(self.model_args)
        reward_funcs = list(getattr(self.script_args, "reward_funcs", []))
        reward_weights = getattr(self.training_args, "reward_weights", None)
        dataset_binding_mode, dataset_binding = self._dataset_binding()

        semantic_override_receipt = None
        if dataset_binding_mode == "chk1_semantic_override":
            semantic_override_receipt = {
                "scope": {
                    "stage": dataset_binding["dataset_semantic_override_stage"],
                    "operation": "sft_training",
                    "downstream_stages_allowed": [],
                },
                "candidate_manifest": {
                    "path": dataset_binding[
                        "dataset_semantic_override_candidate_manifest"
                    ],
                    "sha256": dataset_binding[
                        "dataset_semantic_override_candidate_manifest_sha256"
                    ],
                },
                "deterministic_validation_receipt": {
                    "path": dataset_binding[
                        "dataset_semantic_override_validation_receipt"
                    ],
                    "sha256": dataset_binding[
                        "dataset_semantic_override_validation_receipt_sha256"
                    ],
                },
                "failed_semantic_audit_summary": {
                    "path": dataset_binding["dataset_semantic_override_audit_summary"],
                    "sha256": dataset_binding[
                        "dataset_semantic_override_audit_summary_sha256"
                    ],
                },
                "authorization_receipt": {
                    "path": dataset_binding[
                        "dataset_semantic_override_authorization_receipt"
                    ],
                    "sha256": dataset_binding[
                        "dataset_semantic_override_authorization_receipt_sha256"
                    ],
                },
            }

        standalone_chk3_receipt = None
        if dataset_binding_mode == "standalone_chk3_direct_non_promotable":
            standalone_chk3_receipt = {
                "schema_version": STANDALONE_CHK3_BINDING_SCHEMA,
                "scope": {
                    "scope_id": dataset_binding["dataset_standalone_chk3_scope"],
                    "training_stage": "chk3",
                    "operation": "direct_sft_training",
                    "parent_stage": "chk1",
                    "canonical_dag_bindable": False,
                    "promotable_as_canonical_chk3": False,
                    "downstream_stages_allowed": [],
                },
                "release_manifest": {
                    "path": dataset_binding["dataset_standalone_chk3_release_manifest"],
                    "sha256": dataset_binding[
                        "dataset_standalone_chk3_release_manifest_sha256"
                    ],
                },
            }
            if (
                standalone_chk3_receipt["scope"]["scope_id"]
                != STANDALONE_CHK3_DIRECT_SCOPE
            ):
                raise ValueError(
                    "standalone chk3 direct-SFT scope is not the supported "
                    "versioned non-promotable scope"
                )

        chk4_decision_receipt = None
        fixed_schedule_receipt = None
        if dataset_binding_mode == "chk4_decision_release":
            verified_chk4 = self._verify_chk4_release_binding(dataset_binding)
            chk4_decision_receipt = {
                "schema_version": verified_chk4["schema_version"],
                "scope": {
                    "role": dataset_binding["dataset_chk4_role"],
                    "canonical_dag_bindable": False,
                    "test_is_sealed_evaluation_only": True,
                },
                "release_manifest": {
                    "path": dataset_binding["dataset_chk4_release_manifest"],
                    "sha256": dataset_binding["dataset_chk4_release_manifest_sha256"],
                },
                "system_prompt_sha256": CHK4_STUDENT_SYSTEM_PROMPT_SHA256,
                "tokenizer_bundle": verified_chk4["tokenizer_binding"],
            }
            if (
                getattr(self.training_args, "train_sampler", "default")
                in (
                    *FIXED_SCHEDULE_SAMPLER_TYPES,
                    CORRECTION_FIXED_SCHEDULE_SAMPLER_TYPE,
                )
            ):
                if (
                    self.training_args.train_sampler
                    == CORRECTION_FIXED_SCHEDULE_SAMPLER_TYPE
                ):
                    fixed_schedule_receipt = (
                        validate_correction_fixed_schedule_runtime(
                            training_args=self.training_args,
                            release_binding=verified_chk4,
                        )
                    )
                else:
                    fixed_schedule_receipt = validate_fixed_schedule_runtime(
                        training_args=self.training_args,
                        release_binding=verified_chk4,
                    )

        chk4_branch_launch = None
        chk4_branch_environment = {
            "profile": os.environ.get("FOMC_CHK4_BRANCH_PROFILE"),
            "branch_id": os.environ.get("FOMC_CHK4_BRANCH_ID"),
            "stage": os.environ.get("FOMC_CHK4_BRANCH_STAGE"),
            "config_path": os.environ.get("FOMC_CHK4_BRANCH_CONFIG_PATH"),
            "config_sha256": os.environ.get("FOMC_CHK4_BRANCH_CONFIG_SHA256"),
        }
        required_chk4_branch_fields = {
            "branch_id",
            "stage",
            "config_path",
            "config_sha256",
        }
        configured_chk4_branch_fields = {
            key
            for key in required_chk4_branch_fields
            if chk4_branch_environment.get(key)
        }
        if configured_chk4_branch_fields or chk4_branch_environment["profile"]:
            if configured_chk4_branch_fields != required_chk4_branch_fields:
                missing = sorted(
                    required_chk4_branch_fields - configured_chk4_branch_fields
                )
                raise ValueError(
                    "partial chk4 standalone branch launch binding; missing "
                    + ", ".join(missing)
                )
            config_path = Path(chk4_branch_environment["config_path"]).resolve()
            if not config_path.is_file() or config_path.is_symlink():
                raise ValueError(
                    f"chk4 branch training config is missing or unsafe: {config_path}"
                )
            config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
            if config_sha256 != chk4_branch_environment["config_sha256"]:
                raise ValueError("chk4 branch training config hash drift")
            chk4_branch_launch = {
                "branch_id": chk4_branch_environment["branch_id"],
                "stage": chk4_branch_environment["stage"],
                "config": {
                    "path": str(config_path),
                    "sha256": config_sha256,
                },
            }
            if chk4_branch_environment["profile"]:
                chk4_branch_launch["profile"] = chk4_branch_environment["profile"]

        runtime_config = {
            "schema_version": 1,
            "chk4_standalone_branch": chk4_branch_launch,
            "model": {
                "model_name_or_path": self.model_args.model_name_or_path,
                "model_revision": getattr(self.model_args, "model_revision", None),
                "torch_dtype": getattr(
                    self.model_args,
                    "torch_dtype",
                    getattr(self.model_args, "dtype", None),
                ),
                "dtype": getattr(
                    self.model_args,
                    "dtype",
                    getattr(self.model_args, "torch_dtype", None),
                ),
                "attn_implementation": getattr(
                    self.model_args, "attn_implementation", None
                ),
                "quantization": {
                    "enabled": quantization_config is not None,
                    "load_in_4bit": bool(
                        getattr(self.model_args, "load_in_4bit", False)
                    ),
                    "load_in_8bit": bool(
                        getattr(self.model_args, "load_in_8bit", False)
                    ),
                    "bnb_4bit_quant_type": getattr(
                        self.model_args, "bnb_4bit_quant_type", None
                    ),
                    "bnb_4bit_compute_dtype": str(
                        getattr(
                            self.model_args,
                            "bnb_4bit_compute_dtype",
                            None,
                        )
                        or getattr(self.model_args, "dtype", None)
                    ),
                    "bnb_4bit_quant_storage": str(
                        getattr(self.model_args, "bnb_4bit_quant_storage", None)
                    ),
                    "bnb_4bit_use_double_quant": bool(
                        getattr(self.model_args, "use_bnb_nested_quant", False)
                    ),
                    "prepared_for_kbit_training": quantization_config is not None,
                },
            },
            "dataset": {
                "name": getattr(self.script_args, "dataset_name", None),
                "binding_mode": dataset_binding_mode,
                "prompt_column": getattr(
                    self.script_args, "dataset_prompt_column", None
                ),
                "train_split": getattr(self.script_args, "dataset_train_split", None),
                "eval_split": getattr(self.script_args, "dataset_test_split", None),
                "release_manifest": getattr(
                    self.script_args, "dataset_release_manifest", None
                ),
                "release_manifest_sha256": getattr(
                    self.script_args, "dataset_release_manifest_sha256", None
                ),
                "standalone_chk3_direct": standalone_chk3_receipt,
                "chk4_decision": chk4_decision_receipt,
                "semantic_override": semantic_override_receipt,
                "user_prompt_suffix_sha256": (
                    hashlib.sha256(
                        self.script_args.user_prompt_suffix.encode("utf-8")
                    ).hexdigest()
                    if getattr(self.script_args, "user_prompt_suffix", None)
                    else None
                ),
            },
            "training": {
                "output_dir": self.training_args.output_dir,
                "learning_rate": getattr(self.training_args, "learning_rate", None),
                "num_train_epochs": getattr(
                    self.training_args, "num_train_epochs", None
                ),
                "max_steps": getattr(self.training_args, "max_steps", None),
                "optimizer": getattr(self.training_args, "optim", None),
                "lr_scheduler_type": getattr(
                    self.training_args, "lr_scheduler_type", None
                ),
                "warmup_ratio": getattr(self.training_args, "warmup_ratio", None),
                "warmup_steps": getattr(self.training_args, "warmup_steps", None),
                "gradient_accumulation_steps": getattr(
                    self.training_args, "gradient_accumulation_steps", None
                ),
                "gradient_checkpointing": getattr(
                    self.training_args, "gradient_checkpointing", None
                ),
                "per_device_train_batch_size": getattr(
                    self.training_args, "per_device_train_batch_size", None
                ),
                "per_device_eval_batch_size": getattr(
                    self.training_args, "per_device_eval_batch_size", None
                ),
                "seed": getattr(self.training_args, "seed", None),
                "bf16": getattr(self.training_args, "bf16", None),
            },
            "training_sampler": fixed_schedule_receipt,
            "generation": {
                "max_prompt_length": getattr(
                    self.training_args, "max_prompt_length", None
                ),
                "max_completion_length": getattr(
                    self.training_args, "max_completion_length", None
                ),
                "num_generations": getattr(self.training_args, "num_generations", None),
                "temperature": getattr(self.training_args, "temperature", None),
                "top_p": getattr(self.training_args, "top_p", None),
            },
            "peft": {
                "merged_model_path": getattr(
                    self.peft_args, "peft_merged_model_path", None
                ),
                "r": getattr(self.peft_args, "peft_r", None),
                "lora_alpha": getattr(self.peft_args, "peft_lora_alpha", None),
                "lora_dropout": getattr(self.peft_args, "peft_lora_dropout", None),
                "bias": getattr(self.peft_args, "peft_bias", None),
                "target_modules": getattr(self.peft_args, "peft_target_modules", None),
            },
            "rewards": {
                "reward_funcs": reward_funcs,
                "reward_weights": reward_weights,
            },
            "environment": {
                "device": str(getattr(self.training_args, "device", "")),
                "n_gpu": getattr(self.training_args, "n_gpu", None),
                "world_size": getattr(self.training_args, "world_size", None),
                "process_index": getattr(self.training_args, "process_index", None),
                "local_process_index": getattr(
                    self.training_args, "local_process_index", None
                ),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_names": [
                    torch.cuda.get_device_name(idx)
                    for idx in range(torch.cuda.device_count())
                ],
            },
        }

        if {
            "answer",
            "reasoning",
            "online",
            "grounded_analysis_v2",
            "grounded_analysis_v3",
            "grounded_analysis_v3_deepseek_high",
            "grounded_analysis_v3_deepseek_low",
            "grounded_analysis_v3_deepseek_low_timeout600",
        } & set(reward_funcs):
            runtime_config["judge"] = get_online_reward_settings(
                url=getattr(self.script_args, "judge_url", None),
                model=getattr(self.script_args, "judge_model", None),
                timeout=getattr(self.script_args, "judge_timeout", None),
                verbose=getattr(self.script_args, "judge_verbose", None),
                sleep_seconds=getattr(self.script_args, "judge_sleep_seconds", None),
                api_key_env=getattr(self.script_args, "judge_api_key_env", None),
            )
            # Runtime provenance must never persist credential contents.
            runtime_config["judge"].pop("api_key", None)
            runtime_config["judge"].update(
                {
                    "max_retries": getattr(self.script_args, "judge_max_retries", None),
                    "backoff_seconds": getattr(
                        self.script_args, "judge_backoff_seconds", None
                    ),
                    "tokenizer_path": getattr(
                        self.script_args, "judge_tokenizer_path", None
                    ),
                    "max_model_len": getattr(
                        self.script_args, "judge_max_model_len", None
                    ),
                    "max_completion_tokens": getattr(
                        self.script_args, "judge_max_completion_tokens", None
                    ),
                    "candidate_reserve_tokens": getattr(
                        self.script_args, "judge_candidate_reserve_tokens", None
                    ),
                    "boundary_margin_tokens": getattr(
                        self.script_args, "judge_boundary_margin_tokens", None
                    ),
                }
            )

        return runtime_config

    def save_runtime_config(self):
        is_world_process_zero = (
            int(getattr(self.training_args, "process_index", 0)) == 0
        )
        if is_world_process_zero:
            runtime_config = self._build_runtime_config()
            os.makedirs(self.training_args.output_dir, exist_ok=True)
            destination = self._runtime_config_path()
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{os.path.basename(destination)}.",
                suffix=".tmp",
                dir=self.training_args.output_dir,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(runtime_config, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, destination)
            finally:
                if os.path.exists(temporary_path):
                    os.unlink(temporary_path)
            self.logger.info("✅ Runtime configuration saved to %s", destination)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def start_train(self):
        os.makedirs(self.training_args.output_dir, exist_ok=True)
        last_checkpoint = self.load_checkpoint(self.training_args.output_dir)
        self.save_runtime_config()

        self.logger.info("*** 🚀 Start Training ***")
        checkpoint = None
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(
            self.dataset[self.script_args.dataset_train_split]
        )
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
            metrics["eval_samples"] = len(
                self.dataset[self.script_args.dataset_test_split]
            )
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
            loss_jsonl = os.path.join(
                self.training_args.output_dir, "loss_history.jsonl"
            )
            save_plot = os.path.join(
                self.training_args.output_dir, "training_curve.png"
            )
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

        self.logger.info(
            f"✅ Merged model saved to {self.peft_args.peft_merged_model_path}"
        )
