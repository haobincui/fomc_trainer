import importlib.util
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer  # type: ignore

from trl import get_quantization_config


from ..configs import GRPOConfig, ModelConfig, SFTConfig


logger = logging.getLogger(__name__)


def _resolve_model_dtype(model_args: ModelConfig):
    dtype_name = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    return dtype_name if dtype_name in ["auto", None] else getattr(torch, dtype_name)  # type: ignore[arg-type]


def _resolve_attn_implementation(model_args: ModelConfig):
    attn_implementation = model_args.attn_implementation
    if attn_implementation == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        logger.warning("flash-attn is not installed; falling back from flash_attention_2 to sdpa")
        attn_implementation = "sdpa"
        model_args.attn_implementation = attn_implementation
    return attn_implementation


def get_tokenizer(
    model_args: ModelConfig, training_args: SFTConfig | GRPOConfig
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(
    model_args: ModelConfig, training_args: SFTConfig | GRPOConfig
) -> AutoModelForCausalLM:
    """Get the model"""
    torch_dtype = _resolve_model_dtype(model_args)
    quantization_config = get_quantization_config(model_args)
    use_cache = False if training_args.gradient_checkpointing else True

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=_resolve_attn_implementation(model_args),
        torch_dtype=torch_dtype,
        device_map=None,
    )
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "use_cache"):
        model.config.text_config.use_cache = use_cache
    return model


