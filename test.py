from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "Qwen/Qwen2.5-3B-Instruct"

# model = AutoModelForCausalLM.from_pretrained("../LLMs/qwen2.5-3b").to("cuda")
# tokenizer = AutoTokenizer.from_pretrained("../LLMs/qwen2.5-3b")
# model = AutoModelForCausalLM.from_pretrained("../LLMs/qwen2.5-3b-instruction").to("cuda")
# tokenizer = AutoTokenizer.from_pretrained("../LLMs/qwen2.5-3b-instruction")
model = AutoModelForCausalLM.from_pretrained("../simple_r1_train/outputs/Qwen-3B-GRPO/checkpoint-467").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("../simple_r1_train/outputs/Qwen-3B-GRPO/checkpoint-467")
# model = AutoModelForCausalLM.from_pretrained("../simple_r1_train/outputs/Qwen-3B-GRPO/checkpoint-200").to("cuda")
# tokenizer = AutoTokenizer.from_pretrained("../simple_r1_train/outputs/Qwen-3B-GRPO/checkpoint-200")

prompt = "If 𝑎 > 1, then the sum of the real solutions of√︁ 𝑎 − √𝑎 + 𝑥 = 𝑥 is equal to"
messages = [
    {"role": "system", "content": """ Respond in the following format: <think> ... </think> <answer> ... </answer> """},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)