from transformers import AutoTokenizer, AutoModelForCausalLM

cache_dir = "kanana"

tokenizer = AutoTokenizer.from_pretrained(
    "kakaocorp/kanana-1.5-2.1b-instruct-2505",
    cache_dir=cache_dir,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "kakaocorp/kanana-1.5-2.1b-instruct-2505",
    cache_dir=cache_dir,
    trust_remote_code=True
)

messages = [{"role": "user", "content": "Who are you?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
