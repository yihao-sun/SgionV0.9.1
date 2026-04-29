#!/usr/bin/env python3
"""
Qwen2.5-1.5B 模型 QLoRA 微调脚本
运行方式：python scripts/finetune_qwen.py
"""

import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from transformers import Trainer

# 配置参数
MODEL_NAME = "models/Qwen2.5-1.5B"
DATASET_PATH = "data/ee_finetune_dataset.jsonl"
OUTPUT_DIR = "output/qwen2.5-1.5b-ee"

# QLoRA 配置
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# 训练配置
TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
MAX_STEPS = 1000
WARMUP_STEPS = 100

# 加载数据集
def load_dataset():
    samples = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            # 构建训练样本格式
            text = f"### Instruction:\n{data['instruction']}\n\n### Response:\n{data['output']}\n"
            samples.append(text)
    
    return samples

# 主函数
def main():
    # 加载分词器
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    samples = load_dataset()
    print(f"Dataset loaded with {len(samples)} samples")
    
    # 标记化数据
    def tokenize_function(examples):
        return tokenizer(examples, padding="max_length", truncation=True, max_length=512)
    
    tokenized_samples = tokenize_function(samples)
    dataset = Dataset.from_dict({
        "input_ids": tokenized_samples["input_ids"],
        "attention_mask": tokenized_samples["attention_mask"],
        "labels": tokenized_samples["input_ids"].copy()
    })
    
    # 配置 BitsAndBytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )
    
    # 加载模型
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # 配置 LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 创建 PEFT 模型
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        warmup_steps=WARMUP_STEPS,
        logging_steps=100,
        save_steps=500,
        bf16=True,
        optim="paged_adamw_32bit",
        save_total_limit=3,
        push_to_hub=False,
        remove_unused_columns=False
    )
    
    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存模型
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Training completed! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()