from transformers import AutoProcessor, AutoModelForCausalLM
import torch

MODEL_ID = "models/gemma-4-E4B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="cuda:0",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain why falling inflation does not always imply immediate rate cuts."},
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,   # 关键就在这里
)

inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

parsed = processor.parse_response(response)
print(response)
# print("THINKING:\n", parsed["thinking"])
# print("ANSWER:\n", parsed["answer"])