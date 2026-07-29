
_SYSTEM_PROMPT = """<|think|>
You are a helpful AI Assistant, designed to provide well-reasoned and detailed responses.
Think carefully as internal reasoning before answering.
Use the model's native thought-channel output format and do not emit legacy XML wrapper tags.
"""

_MODEL = None
_MODEL_CONFIG = None


def get_system_prompt() -> str:
    """Return the exact system prompt used by the generation wrapper."""

    return _SYSTEM_PROMPT


def load_model(
    model_path,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=256,
    max_model_len=16384,
    tokenizer_path=None,
):
    global _MODEL, _MODEL_CONFIG

    requested_config = (
        model_path,
        float(temperature),
        float(top_p),
        int(max_new_tokens),
        int(max_model_len),
        tokenizer_path,
    )
    if _MODEL is None or _MODEL_CONFIG != requested_config:
        # Keep vLLM optional for data-processing and unit-test imports.
        from open_r1.generation_model.model import Model

        print(f"🚀 Loading model from: {model_path} ...")
        _MODEL = Model(
            model_path=model_path,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            max_model_len=max_model_len,
            tokenizer_path=tokenizer_path,
        )
        _MODEL_CONFIG = requested_config
    else:
        print("🚀 Using cached model ...")

    return _MODEL


def generate_response(prompt, model_path, **kwargs):
    seed = kwargs.pop("seed", None)
    model = load_model(model_path, **kwargs)

    message = [{"role": "system", "content": _SYSTEM_PROMPT},
               {"role": "user", "content": prompt}]
    
    return model.chat_completion(message, seed=seed)


def generate_responses(prompts, model_path, **kwargs):
    seed = kwargs.pop("seed", None)
    row_seeds = kwargs.pop("row_seeds", None)
    return_metadata = bool(kwargs.pop("return_metadata", False))
    model = kwargs.pop("model", None) or load_model(model_path, **kwargs)
    messages = []
    for prompt in prompts:
        message = [{"role": "system", "content": _SYSTEM_PROMPT},
                   {"role": "user", "content": prompt}]
        messages.append(message)
    responses = model.batch_chat_completion(
        messages,
        seed=seed,
        seeds=row_seeds,
        return_metadata=return_metadata,
    )

    return responses
