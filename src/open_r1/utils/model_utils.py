import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer # type: ignore

from trl import ModelConfig, get_kbit_device_map, get_quantization_config, get_peft_config


from ..configs import GRPOConfig, SFTConfig
from ..generation_model.model import Model


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
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype) # type: ignore
    )
    quantization_config = get_quantization_config(model_args)


    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        # device_map=get_kbit_device_map() if quantization_config is not None else None,
        device_map = None,
    )
    if quantization_config:
        model_kwargs["quantization_config"] 
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model



def load_generation_model(model_path: str, temperature: float, top_p: float, max_new_tokens: int) -> Model:
    # os.environ["CUDA_VISIBLE_DEVICES"] = 0, 1
    print(f"🚀 Loading model ...")

    return Model(
        model_path=model_path,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

