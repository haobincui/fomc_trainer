
from open_r1.generation_model.model import Model

_SYSTEM_PROMPT = """
  You are a helpful AI Assistant, designed to provide well-reasoned and detailed responses. 
  You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. 
  The reasoning process MUST BE enclosed within <think> and </think> tags. The answer MUST BE enclosed within <answer> and </answer> tags.
"""

_MODEL = None

_MODEL = None

def load_model(model_path, temperature=0.7, top_p=0.9, max_new_tokens=256):
    global _MODEL
    if _MODEL is None:
        print(f"🚀 Loading model from {model_path} ...")
        _MODEL = Model(
            model_path=model_path,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
    else:
        print("🚀 Using cached model ...")
    return _MODEL


def generate_response(prompt, model_path, **kwargs):
    model = load_model(model_path, **kwargs)

    message = [{"role": "system", "content": _SYSTEM_PROMPT},
               {"role": "user", "content": prompt}]
    
    return model.chat_completion(message)


def generate_responses(prompts, model_path, **kwargs):
    model = load_model(model_path, **kwargs)
    results = []

    for prompt in prompts:
        message = [{"role": "system", "content": _SYSTEM_PROMPT},
                   {"role": "user", "content": prompt}]
        response = model.chat_completion(message)
        results.append(response)

    return results









