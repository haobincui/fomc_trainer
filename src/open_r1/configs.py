# coding=utf-8
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

from dataclasses import dataclass, field
from typing import Optional

import trl


@dataclass
class ModelConfig(trl.ModelConfig):
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={"help": "Backward-compatible alias for `dtype`."},
    )
    use_cache: bool = field(
        default=True,
        metadata={
            "help": (
                "Backward-compatible model flag accepted from retrain configs; "
                "gradient checkpointing remains authoritative at runtime."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.torch_dtype is not None:
            if self.dtype not in (None, "float32", self.torch_dtype):
                raise ValueError(
                    f"Conflicting dtype values: dtype={self.dtype}, torch_dtype={self.torch_dtype}"
                )
            self.dtype = self.torch_dtype
        self.torch_dtype = self.dtype


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    checkpoint_keep_last: int = field(
        default=0,
        metadata={
            "help": (
                "Number of most recent checkpoints retained by the optional "
                "checkpoint_retention callback. Zero disables the setting."
            )
        },
    )
    checkpoint_keep_every_n_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Retain every Nth checkpoint with the optional "
                "checkpoint_retention callback. Zero disables the setting."
            )
        },
    )
    checkpoint_keep_steps: list[int] = field(
        default_factory=list,
        metadata={"help": "Additional exact checkpoint steps to retain permanently."},
    )
    chat_template: Optional[str] = field(
        default=None, metadata={"help": "The chat template to use."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to overwrite the Hub revision."}
    )
    push_to_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to push to a Hub revision/branch."}
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Backward-compatible prompt truncation length for GRPO prompt encoding."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Backward-compatible no-op flag accepted from legacy configs."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.max_prompt_length is not None and hasattr(self, "chat_template_kwargs"):
            chat_template_kwargs = getattr(self, "chat_template_kwargs", None) or {}
            chat_template_kwargs.setdefault("truncation", True)
            chat_template_kwargs.setdefault("max_length", self.max_prompt_length)
            self.chat_template_kwargs = chat_template_kwargs


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    checkpoint_keep_last: int = field(
        default=0,
        metadata={
            "help": (
                "Number of most recent checkpoints retained by the optional "
                "checkpoint_retention callback. Zero disables the setting."
            )
        },
    )
    checkpoint_keep_every_n_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Retain every Nth checkpoint with the optional "
                "checkpoint_retention callback. Zero disables the setting."
            )
        },
    )
    checkpoint_keep_steps: list[int] = field(
        default_factory=list,
        metadata={"help": "Additional exact checkpoint steps to retain permanently."},
    )
    train_sampler: str = field(
        default="default",
        metadata={
            "help": (
                "Training sampler contract. 'default' preserves Trainer behavior; "
                "'manifest_fixed_schedule_v1' consumes the fixed 192-row chk4 "
                "schedule; 'manifest_fixed_schedule_v2' consumes a manifest-bound "
                "dynamic-length schedule; 'manifest_fixed_schedule_v3' consumes "
                "the independently versioned 48-row correction schedule. All are "
                "sequential and forbid secondary shuffling."
            )
        },
    )
    chat_template: Optional[str] = field(
        default=None, metadata={"help": "The chat template to use."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to overwrite the Hub revision."}
    )
    push_to_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to push to a Hub revision/branch."}
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Backward-compatible alias for `max_length` in legacy SFT configs."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Backward-compatible no-op flag accepted from legacy configs."
        },
    )

    # deepspeed: str = Optional[field](
    #     default=None,
    #     metadata={"help": "deepseek config path"}
    # )

    def __post_init__(self):
        super().__post_init__()
        if self.train_sampler not in {
            "default",
            "manifest_fixed_schedule_v1",
            "manifest_fixed_schedule_v2",
            "manifest_fixed_schedule_v3",
        }:
            raise ValueError(
                "train_sampler must be default, manifest_fixed_schedule_v1, "
                "manifest_fixed_schedule_v2, or manifest_fixed_schedule_v3"
            )
        if self.max_prompt_length is not None:
            if getattr(self, "max_length", None) in (
                None,
                trl.SFTConfig.__dataclass_fields__["max_length"].default,
            ):
                self.max_length = self.max_prompt_length


@dataclass
class GRPOScriptArguments(trl.ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'ioi_code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    user_prompt_suffix: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional audited instruction appended to each GRPO user prompt "
                "before chat-template rendering."
            )
        },
    )
    save_reward: bool = field(
        default=False,
        metadata={
            "help": "Whether to save the reward values to a file. If True, the reward values will be saved to a file."
        },
    )
    judge_url: Optional[str] = field(
        default=None,
        metadata={"help": "Optional override for the online judge HTTP endpoint."},
    )
    judge_model: Optional[str] = field(
        default=None,
        metadata={"help": "Optional override for the online judge model name."},
    )
    judge_tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Immutable local tokenizer path for judge context gates."},
    )
    judge_max_model_len: int = field(
        default=6144,
        metadata={"help": "Immutable judge server context length."},
    )
    judge_max_completion_tokens: int = field(
        default=512,
        metadata={"help": "Tokens reserved for the structured judge response."},
    )
    judge_candidate_reserve_tokens: int = field(
        default=1536,
        metadata={"help": "Conservative Qwen-token candidate reserve for preflight."},
    )
    judge_boundary_margin_tokens: int = field(
        default=32,
        metadata={"help": "Non-additive tokenizer boundary margin for preflight."},
    )
    judge_timeout: int = field(
        default=180,
        metadata={"help": "Timeout in seconds for each online judge request."},
    )
    judge_max_retries: int = field(
        default=3,
        metadata={"help": "Fail-closed judge attempts per scored completion (1-5)."},
    )
    judge_backoff_seconds: float = field(
        default=1.0,
        metadata={"help": "Initial exponential judge retry backoff in seconds (0-10)."},
    )
    judge_sleep_seconds: float = field(
        default=0.0,
        metadata={"help": "Optional sleep interval between online judge requests."},
    )
    judge_verbose: bool = field(
        default=False,
        metadata={
            "help": "Whether to print full online judge responses during training."
        },
    )
    judge_api_key_env: Optional[str] = field(
        default="OPEN_R1_JUDGE_API_KEY",
        metadata={
            "help": "Environment variable name used to resolve the judge API key."
        },
    )

    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={
            "help": "Maximum (negative) penalty for for repetition penalty reward"
        },
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash", "cpp"],
        },
    )
    code_eval_test_batch_size: int = field(
        default=1,
        metadata={
            "help": "for each generation, evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases. Useful to avoid overloading the eval server + save time on wrong solutions"
        },
    )
    parallel_code_exec_per_proc: int = field(
        default=2,
        metadata={
            "help": "Number of parallel E2B code executions per process. Default of 2 is suitable for the Free Hobby tier of E2B with 8 GPUs used for training."
        },
    )

    dataset_name: str = field(
        default=None,
        metadata={"help": "Path to the dataset or dataset name."},
    )

    dataset_prompt_column: Optional[str] = field(
        default="prompt", metadata={"help": "Column to use as prompts for training."}
    )

    dataset_train_split: str = field(
        default="train",
        metadata={"help": "Split to use for training."},
    )

    dataset_test_split: str = field(
        default="validation",
        metadata={"help": "Split to use for evaluation."},
    )

    dataset_chk4_role: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Role selected from an immutable chk4 Decision release: "
                "decision_sft or decision_grpo."
            )
        },
    )

    dataset_chk4_release_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Immutable chk4 Decision release manifest."},
    )

    dataset_chk4_release_manifest_sha256: Optional[str] = field(
        default=None,
        metadata={"help": "Externally pinned SHA-256 of the chk4 Decision manifest."},
    )

    e2b_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL for the E2B router. See scripts/e2b_router.py"},
    )

    morph_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL for the MorphCloud router. See scripts/morph_router.py"},
    )

    code_provider: Optional[str] = field(
        default="e2b",
        metadata={
            "help": "Provider for code execution. Options: 'e2b', 'local', 'morph'.",
            "choices": ["e2b", "local", "morph"],
        },
    )

    ioi_provider: Optional[str] = field(
        default="piston",
        metadata={
            "help": "Provider for IOI code execution. Options: 'piston', 'morph'.",
            "choices": ["piston", "morph"],
        },
    )


@dataclass
class SFTScriptArguments(trl.ScriptArguments):
    """
    Script arguments for the SFT training script.

    Args:
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={
            "help": "Maximum (negative) penalty for for repetition penalty reward"
        },
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash", "cpp"],
        },
    )
    code_eval_test_batch_size: int = field(
        default=1,
        metadata={
            "help": "for each generation, evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases. Useful to avoid overloading the eval server + save time on wrong solutions"
        },
    )
    parallel_code_exec_per_proc: int = field(
        default=2,
        metadata={
            "help": "Number of parallel E2B code executions per process. Default of 2 is suitable for the Free Hobby tier of E2B with 8 GPUs used for training."
        },
    )

    dataset_name: str = field(
        default=None,
        metadata={"help": "Path to the dataset or dataset name."},
    )

    dataset_prompt_column: str = field(
        default="prompt",
        metadata={"help": "Column to use as prompts for training."},
    )

    dataset_train_split: str = field(
        default="train",
        metadata={"help": "Split to use for training."},
    )

    dataset_test_split: str = field(
        default="validation",
        metadata={"help": "Split to use for evaluation."},
    )

    dataset_chk4_role: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Role selected from an immutable chk4 Decision release: "
                "decision_sft or decision_grpo."
            )
        },
    )

    dataset_chk4_release_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Immutable chk4 Decision release manifest."},
    )

    dataset_chk4_release_manifest_sha256: Optional[str] = field(
        default=None,
        metadata={"help": "Externally pinned SHA-256 of the chk4 Decision manifest."},
    )

    dataset_release_manifest: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional immutable clean-SFT release manifest. When set, the "
                "training loader verifies the sealed split files before loading."
            )
        },
    )

    dataset_release_manifest_sha256: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Lowercase SHA-256 of dataset_release_manifest; must be supplied "
                "together with the manifest path."
            )
        },
    )

    dataset_standalone_chk3_scope: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Explicit standalone chk1-to-chk3 direct-SFT scope. Only the "
                "versioned non-promotable scope is accepted by the trainer."
            )
        },
    )

    dataset_standalone_chk3_release_manifest: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Immutable standalone chk3 Minutes-SFT release manifest. This "
                "binding never authorizes canonical DAG promotion."
            )
        },
    )

    dataset_standalone_chk3_release_manifest_sha256: Optional[str] = field(
        default=None,
        metadata={"help": "Pinned SHA-256 of the standalone chk3 release manifest."},
    )

    dataset_paper_chk2_scope: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Explicit non-DAG paper chk2 Minutes-SFT training scope. This "
                "binding is separate from the canonical chk2 analysis-GRPO stage."
            )
        },
    )

    dataset_paper_chk2_release_manifest: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Immutable paper chk2 recovery release manifest. The runtime "
                "loads only its sealed train and validation splits."
            )
        },
    )

    dataset_paper_chk2_release_manifest_sha256: Optional[str] = field(
        default=None,
        metadata={
            "help": "Externally pinned file SHA-256 of the paper chk2 release manifest."
        },
    )

    dataset_semantic_override_stage: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Explicit stage for the exceptional semantic-audit override. "
                "Only the literal value 'chk1' is accepted."
            )
        },
    )

    dataset_semantic_override_candidate_manifest: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Immutable pending clean-SFT candidate manifest used only by "
                "the explicitly authorized chk1 semantic override."
            )
        },
    )

    dataset_semantic_override_candidate_manifest_sha256: Optional[str] = field(
        default=None,
        metadata={"help": "Pinned SHA-256 of the chk1 override candidate manifest."},
    )

    dataset_semantic_override_validation_receipt: Optional[str] = field(
        default=None,
        metadata={
            "help": "Passed deterministic clean-SFT validation receipt for the override."
        },
    )

    dataset_semantic_override_validation_receipt_sha256: Optional[str] = field(
        default=None,
        metadata={"help": "Pinned SHA-256 of the deterministic validation receipt."},
    )

    dataset_semantic_override_audit_summary: Optional[str] = field(
        default=None,
        metadata={
            "help": "Failed source-only semantic-audit summary acknowledged by the override."
        },
    )

    dataset_semantic_override_audit_summary_sha256: Optional[str] = field(
        default=None,
        metadata={"help": "Pinned SHA-256 of the failed semantic-audit summary."},
    )

    dataset_semantic_override_authorization_receipt: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Hash-bound explicit-user authorization receipt restricted to "
                "chk1 SFT and prohibiting downstream stages."
            )
        },
    )

    dataset_semantic_override_authorization_receipt_sha256: Optional[str] = field(
        default=None,
        metadata={"help": "Pinned SHA-256 of the chk1-only authorization receipt."},
    )

    e2b_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL for the E2B router. See scripts/e2b_router.py"},
    )

    morph_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL for the MorphCloud router. See scripts/morph_router.py"},
    )

    code_provider: Optional[str] = field(
        default="e2b",
        metadata={
            "help": "Provider for code execution. Options: 'e2b', 'local', 'morph'.",
            "choices": ["e2b", "local", "morph"],
        },
    )

    ioi_provider: Optional[str] = field(
        default="piston",
        metadata={
            "help": "Provider for IOI code execution. Options: 'piston', 'morph'.",
            "choices": ["piston", "morph"],
        },
    )


@dataclass
class LoraArguments:
    peft_merged_model_path: Optional[str] = field(
        default=None, metadata={"help": "Path to Merged Model."}
    )
    peft_r: int = field(default=8, metadata={"help": "LoRA rank"})
    peft_lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    peft_lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    peft_bias: str = field(
        default="none",
        metadata={
            "help": "LoRA bias mode; retrain-v2 merge verification requires none."
        },
    )

    peft_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={"help": "Target modules to apply LoRA"},
    )
