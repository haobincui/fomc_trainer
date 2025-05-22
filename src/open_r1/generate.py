



from open_r1.utils.model_utils import load_generation_model


_MODEL = None

def generate_response(prompt, model_path = None, temperature=0.7, top_p=0.9, max_new_tokens=256):
    global _MODEL
    if _MODEL is None:
        print(f"🚀 Loading model from {model_path} ...")
        model = load_generation_model(model_path, temperature, top_p, max_new_tokens)
        _MODEL = model
    else:
        print("🚀 Using cached model ...")

    response = _MODEL.generate_completion(prompt)
    return response








