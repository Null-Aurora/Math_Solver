import json
import torch
import re
from tqdm import tqdm
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

test_json_new_path = "test.json"

with open(test_json_new_path, 'r', encoding='utf-8') as file:
    test_data = json.load(file)

model_dir = "./Qwen\Qwen2___5-0___5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    use_fast=False,
    trust_remote_code=True
 )
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="eager"
)
def extract_final_answer(text):
    # 匹配"答案："格式
    colon_match = re.search(r'(?:答案|结果)\s*[:：]?\s*是?\s*([-+]?\d+(?:\.\d+)?)', text)
    if colon_match:
        return colon_match.group(1)

    # 匹配方括号格式
    bracket_match = re.search(r'\[([+-]?\d+\.?\d*)]', text)
    if bracket_match:
        return bracket_match.group(1)

    # 提取最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else "0"
model = PeftModel.from_pretrained(model, model_id="./output/Qwen/checkpoint-3745/")

with open("submit.csv", 'w', encoding='utf-8') as file:
    for idx, row in tqdm(enumerate(test_data)):
        instruction = "这是小学数学1-6年级的校内题目，请逐步分析并解答"
        input_value = row['question']
        id = row['id']

        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]
        full_response = predict(messages, model, tokenizer)
        print(full_response)
        final_answer = extract_final_answer(full_response)
        final_answer = final_answer.replace('\n', ' ').strip()
        print(final_answer)
        file.write(f"{id},{final_answer}\n")


