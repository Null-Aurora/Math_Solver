import json
import re

import torch
from tqdm import tqdm
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

# 配置参数
MODEL_DIR = "./Qwen/Qwen2___5-0___5B-Instruct"
TEST_JSON_PATH = "test.json"
OUTPUT_CSV = "submit.csv"
BATCH_SIZE = 16
MAX_WORKERS = 4


# 初始化模型（全局单例）
def init_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=False,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager"
    )

    # model = PeftModel.from_pretrained(model, model_id="./output/Qwen/checkpoint-3750/")
    return model, tokenizer

def extract_final_answer(text):
    # 匹配"答案是：" "结果是："格式
    colon_match = re.search(r"(?:答案|结果)\s*[:：]?\s*是?\s*([-+]?\d+(?:\.\d+)?)", text)
    if colon_match:
        return colon_match.group(1)

    # 匹配方括号格式
    bracket_match = re.search(r'\[([+-]?\d+\.?\d*)]', text)
    if bracket_match:
        return bracket_match.group(1)

    # 提取最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else "0"


# 批量预测函数
def batch_predict(batch_data, model, tokenizer):
    device = model.device
    batch_texts = []

    # 准备批量输入
    for item in batch_data:
        messages = [
            {"role": "system", "content": "这是小学数学1-6年级的校内题目，让我们一步步思考,然后给出纯数字最终答案"},
            {"role": "user", "content": item["question"]}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        batch_texts.append(text)

    # 批量编码
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # 批量生成
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )

    # 解码结果
    results = []
    for i in range(len(batch_data)):
        output = outputs[i][len(inputs.input_ids[i]):]  # 截取生成部分
        result = tokenizer.decode(output, skip_special_tokens=True).strip()
        print(f"模型原始回答: {result}")
        final_answer = extract_final_answer(result)
        print(f"提取: {final_answer}")
        results.append((batch_data[i]["id"], final_answer))

    return results


# 多线程处理
def process_with_threads(test_data, model, tokenizer):
    result_queue = queue.Queue()
    progress_bar = tqdm(total=len(test_data), desc="Processing")

    def worker(batch):
        try:
            results = batch_predict(batch, model, tokenizer)
            result_queue.put(results)
        except Exception as e:
            print(f"处理失败: {str(e)}")
            result_queue.put([(item["id"], "ERROR") for item in batch])
        finally:
            progress_bar.update(len(batch))

    # 创建线程池
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 分批处理数据
        batches = [test_data[i:i + BATCH_SIZE] for i in range(0, len(test_data), BATCH_SIZE)]
        for batch in batches:
            executor.submit(worker, batch)

    # 收集结果
    all_results = []
    for _ in range(len(batches)):
        all_results.extend(result_queue.get())

    progress_bar.close()
    return sorted(all_results, key=lambda x: int(x[0]))  # 按ID排序


def main():
    # 加载模型
    print("Loading model...")
    model, tokenizer = init_model()

    # 加载测试数据
    with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 多线程处理
    print("Start predicting...")
    results = process_with_threads(test_data, model, tokenizer)

    # 保存结果
    with open(OUTPUT_CSV, 'w', encoding='utf-8') as f:
        for id, response in results:
            f.write(f"{id},{response.replace(',', ' ')}\n")  # 处理CSV中的逗号冲突

    print(f"Done! Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()