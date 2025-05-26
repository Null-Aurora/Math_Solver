import json
import torch
import re
from tqdm import tqdm
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel
from queue import Queue
import threading

# 配置参数
BATCH_SIZE = 64
MAX_WORKERS = 6
MODEL_PATH = "./Qwen/Qwen2___5-0___5B-Instruct"

#
CHECKPOINT_PATH = "./output/Qwen/checkpoint-935/"


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


class PredictWorker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            use_fast=False,
            trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        self.model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
        self.model.eval()
        self.lock = threading.Lock()

    def process_batch(self, batch):
        """批量处理函数"""
        with torch.no_grad(), self.lock:
            try:
                # 准备批量输入
                texts = [
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": "这是小学数学1-6年级的校内题目，请逐步分析并解答"},
                            {"role": "user", "content": item["question"]}
                        ],
                        tokenize=False,
                        add_generation_prompt=True
                    ) for item in batch
                ]

                # 批量编码
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.model.device)

                # 批量生成
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                # 处理结果
                results = []
                for i, item in enumerate(batch):
                    output = outputs[i][len(inputs.input_ids[i]):]
                    response = self.tokenizer.decode(output, skip_special_tokens=True)
                    print(response)
                    final_answer = extract_final_answer(response)
                    results.append((item["id"], final_answer))

                return results
            except Exception as e:
                print(f"批量处理失败: {str(e)}")
                return [(item["id"], "ERROR") for item in batch]


def main():
    # 初始化工作器
    worker = PredictWorker()

    # 加载测试数据
    with open("test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 创建任务队列
    task_queue = Queue()
    batches = [test_data[i:i + BATCH_SIZE] for i in range(0, len(test_data), BATCH_SIZE)]
    for batch in batches:
        task_queue.put(batch)

    # 结果收集
    result_queue = Queue()
    progress = tqdm(total=len(batches), desc="Processing")

    def worker_thread():
        while not task_queue.empty():
            try:
                batch = task_queue.get_nowait()
                results = worker.process_batch(batch)
                result_queue.put(results)
                progress.update(1)
            except:
                break

    # 启动线程池
    threads = []
    for _ in range(MAX_WORKERS):
        t = threading.Thread(target=worker_thread)
        t.start()
        threads.append(t)

    # 等待所有线程完成
    for t in threads:
        t.join()
    progress.close()

    # 收集并排序结果
    all_results = []
    while not result_queue.empty():
        all_results.extend(result_queue.get())
    all_results.sort(key=lambda x: int(x[0]))

    # 写入文件
    with open("submit.csv", "w", encoding="utf-8") as f:
        for id, answer in all_results:
            f.write(f"{id},{answer.replace(',', ' ')}\n")


if __name__ == "__main__":
    main()