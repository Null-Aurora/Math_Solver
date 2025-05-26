import json
import torch
import re
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import swanlab
import os
os.environ["SWANLAB_MODE"] = "disabled"

def process_func(example):
    MAX_LENGTH = 512
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n这是小学数学1-6年级的校内题目，请逐步分析并解答<|im_end|>\n<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant",
        add_special_tokens=False,
    )
    if 'solution_steps' not in example:
        response = tokenizer(f"{example['answer']}", add_special_tokens=False)
    else:
        response = tokenizer(f"{example['solution_steps']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response['input_ids'] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response['input_ids'] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response['input_ids'] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


model_dir = snapshot_download(
    "Qwen/Qwen2.5-0.5B-Instruct",
    cache_dir="./",
    ignore_file_pattern=[".*\.bin"],  # 避免重复下载
    revision="master"
)

# Transformers加载模型权重
# 使用动态路径加载模型
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    use_fast=False,
    trust_remote_code=True,
    padding_side="left"
 )
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
train_json_new_path = "train_with_steps.json"
skipped = 0
with open(train_json_new_path, 'r', encoding='utf-8') as file:
    train_data = json.load(file)
train_dataset = []
for d in train_data:
    if 'solution_steps' not in d or not d['solution_steps']:
        skipped += 1
        print(f"累计已跳过{skipped}道没有步骤的题目")
        continue  # 跳过没有 solution_steps 的样本
    train_dataset.append(process_func(d))

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="./output/Qwen",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=5,
    save_steps=1000,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none"
)

swanlab_callback = SwanLabCallback(
    project="Qwen2.5-0.5B-fintune",
    experiment_name="Qwen/Qwen2.5-0.5B-Instruct",
    config={
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset": "news",
    }
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()
swanlab.finish()

