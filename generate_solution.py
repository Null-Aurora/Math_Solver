import json
import requests
import os
import time
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

#API配置
API_KEY = "sk-W90Imo1xsNjGPSdJ1raR86MvgiOBYUus6393OHDDRfzTXgDA"
API_URL = "https://www.dmxapi.com/v1/chat/completions"
MAX_RETRIES = 3  # 失败重试次数
model = "deepseek-v3"
MAX_WORKERS = 5  # 并发线程数
input_path = "train.json"
output_path = "train_with_steps.jsonl"
lock = threading.Lock()  # 写文件锁

def create_session():
    """创建带重试机制的会话"""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

def call_dmxapi(messages):
    session = create_session()
    headers = {
        'Accept': 'application/json',
        'Authorization': API_KEY,
        'User-Agent': 'DMXAPI/1.0.0 (https://www.dmxapi.com)',
        'Content-Type': 'application/json'
    }
    payload = json.dumps({
        "model": model,
        "messages": messages
    })
    try:
        response = session.post(API_URL, headers=headers, data=payload, timeout=150)
        if response.status_code == 200:
            return response.json()
        print(f"请求失败，状态码：{response.status_code}")
        return None
    except Exception as e:
        print(f"请求异常：{str(e)}")
        return None

def generate_cot_steps(question, answer):
    """生成带解题步骤的思维链"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是一位小学数学老师，需要为题目生成详细解题步骤。要求：\n"
                "1. 使用中文分步骤解释\n"
                "2. 最后用方括号包裹数字答案\n"
                "示例：\n"
                "问题：小明有5个苹果...\n"
                "解答：步骤1...步骤2...答案：[5]"
            )
        },
        {
            "role": "user",
            "content": f"题目：{question}\n正确答案：{answer}"
        }
    ]
    for attempt in range(MAX_RETRIES):
        result = call_dmxapi(messages)
        if result:
            try:
                content = result["choices"][0]["message"]["content"]
                # 验证是否包含答案
                if f"[{answer}]" not in content:
                    print(f"答案验证失败，重试中...（第{attempt + 1}次）")
                    continue
                return content
            except (KeyError, IndexError):
                pass
        time.sleep(1)  # 失败后等待1秒重试

    return f"自动生成失败，请手动补充。正确答案：{answer}"

# 单个任务处理函数（线程安全写入）
def process_item(item):
    item_id = item["id"]
    try:
        item["solution_steps"] = generate_cot_steps(item["question"], item["answer"])
    except Exception as e:
        print(f"处理失败({item_id}): {str(e)}")
        item["solution_steps"] = "生成异常，请手动处理"

    with lock:
        with open(output_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()
    print(f"[完成] {item_id}")


def main():
    # 读取原始数据
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 处理每条数据
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    processed_ids.add(obj["id"])
                except:
                    continue  # 跳过损坏的行
    print(f"已处理 {len(processed_ids)} 条，开始处理剩余 {len(data) - len(processed_ids)} 条。")
    to_process = [item for item in data if item["id"] not in processed_ids]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_item, item) for item in to_process]
        for future in as_completed(futures):
            _ = future.result()  # 捕获异常防止中断

    print("全部处理完成")


if __name__ == "__main__":
    main()