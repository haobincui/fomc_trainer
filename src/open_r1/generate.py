
from open_r1.generation_model.model import Model

_SYSTEM_PROMPT = """<|think|>
You are a helpful AI Assistant, designed to provide well-reasoned and detailed responses.
Think carefully as internal reasoning before answering.
Use the model's native thought-channel output format and do not emit legacy XML wrapper tags.
"""

_MODEL = None
_MODEL_PATH = None

def load_model(model_path, temperature=0.7, top_p=0.9, max_new_tokens=256):
    global _MODEL, _MODEL_PATH

    if _MODEL is None or _MODEL_PATH != model_path:
        print(f"🚀 Loading model from: {model_path} ...")
        _MODEL = Model(
            model_path=model_path,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        _MODEL_PATH = model_path
    else:
        print("🚀 Using cached model ...")

    return _MODEL


def generate_response(prompt, model_path, **kwargs):
    model = load_model(model_path, **kwargs)

    message = [{"role": "system", "content": _SYSTEM_PROMPT},
               {"role": "user", "content": prompt}]
    
    return model.chat_completion(message)


def generate_responses(prompts, model_path, **kwargs):
    
    model = kwargs.pop("model", None) or load_model(model_path, **kwargs)
    results = []
    messages = []
    for prompt in prompts:
        message = [{"role": "system", "content": _SYSTEM_PROMPT},
                   {"role": "user", "content": prompt}]
        messages.append(message)
    responses = model.batch_chat_completion(messages)

    return responses






