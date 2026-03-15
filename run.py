from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = AutoModel.from_pretrained(
    "z-lab/Qwen3-8B-DFlash-b16", 
    trust_remote_code=True, 
    dtype="auto", 
    device_map="cuda:0"
).eval()

target = AutoModelForCausalLM.from_pretrained(
    "/share/public/public_models/Qwen3-8B", 
    dtype="auto", 
    device_map="cuda:0"
).eval()

tokenizer = AutoTokenizer.from_pretrained("/share/public/public_models/Qwen3-8B")
prompt = "How many positive whole-number divisors does 196 have?"
messages = [
    {"role": "user", "content": prompt}
]
# Note: this draft model is used for thinking mode disabled
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generate_ids = model.spec_generate(
    input_ids=model_inputs["input_ids"], 
    max_new_tokens=2048, 
    temperature=0.0, 
    target=target, 
    stop_token_ids=[tokenizer.eos_token_id]
)

print(tokenizer.decode(generate_ids[0], skip_special_tokens=False))