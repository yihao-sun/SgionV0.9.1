#!/usr/bin/env python3
"""
情境渲染器 v2.5 训练脚本
使用扩展特征训练 Qwen 2.5 模型
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import os
import argparse

class ExpressionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据集
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        for item in dataset:
            input_data = item['input']
            output_text = item['output']
            
            # 构建输入状态文本
            coord = input_data['coord']
            valence = input_data['valence']
            arousal = input_data['arousal']
            approach = input_data['approach']
            L = input_data['L']
            stiffness = input_data['stiffness']
            dominant_desire = input_data['dominant_desire']
            goal_type = input_data['goal_type']
            goal_desc = input_data['goal_desc']
            mutual_has_stuck = input_data['mutual_has_stuck']
            mutual_stiffness = input_data['mutual_stiffness']
            
            state_text = (f"坐标:({coord[0]},{coord[1]},{coord[2]}) "
                          f"愉悦度:{valence:.2f} 唤醒度:{arousal:.2f} "
                          f"趋近:{approach:.2f} L:{L} "
                          f"僵化度:{stiffness:.2f} "
                          f"主导欲望:{dominant_desire} "
                          f"内在目标:{goal_type}: {goal_desc} "
                          f"互业状态:{'存在僵化互业' if mutual_has_stuck else '无显著互业执着'}（{mutual_stiffness:.2f}）")
            
            # 构建训练样本
            prompt = f"### 输入状态:\n{state_text}\n\n### 第一人称描述:\n"
            completion = output_text
            
            # 应用 chat template
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            self.data.append(text)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/expression_dataset_v2.json")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", default="models/expression_renderer_lora_v2")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--local_model_path", help="本地模型路径（可选）")
    args = parser.parse_args()
    
    # 选择模型路径
    model_path = args.local_model_path if args.local_model_path and os.path.exists(args.local_model_path) else args.model_name
    print(f"使用模型路径: {model_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    dataset = ExpressionDataset(args.data_path, tokenizer, max_length=args.max_length)
    print(f"数据集大小: {len(dataset)}")
    
    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        optim="paged_adamw_32bit"
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda data: {
            "input_ids": torch.stack([item["input_ids"] for item in data]),
            "attention_mask": torch.stack([item["attention_mask"] for item in data]),
            "labels": torch.stack([item["labels"] for item in data])
        }
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存模型
    print("保存模型...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"模型已保存至: {args.output_dir}")

if __name__ == "__main__":
    main()