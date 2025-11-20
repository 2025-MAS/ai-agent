import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

cache_dir = "kanana"
model_name = "kakaocorp/kanana-1.5-2.1b-instruct-2505"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="auto"
)

messages = [{"role": "user", "content": "누구냐 넌"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        inputs[k] = v.to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
