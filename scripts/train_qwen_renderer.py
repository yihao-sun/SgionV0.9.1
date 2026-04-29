#!/usr/bin/env python3
"""
使用 Qwen2.5-1.5B + LoRA 微调情境渲染器（支持增量训练）
输入：内在状态文本序列（扩展特征）
输出：第一人称意象描述
"""

import json
import torch
import argparse
import os
from functools import partial
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


def format_prompt(inp):
    """将内在状态格式化为 prompt，与训练数据中的输入部分一致（支持扩展特征）"""
    coord = inp['coord']
    state_text = (f"坐标:({coord[0]},{coord[1]},{coord[2]}) "
                  f"愉悦度:{inp['valence']:.2f} 唤醒度:{inp['arousal']:.2f} "
                  f"趋近:{inp['approach']:.2f} L:{inp['L']} "
                  f"僵化度:{inp['stiffness']:.2f}\n"
                  f"主导欲望:{inp['dominant_desire']} "
                  f"内在目标:{inp['goal_type']}:{inp['goal_desc']}\n"
                  f"互业状态:{'有执着' if inp['mutual_has_stuck'] else '无执着'}"
                  f"(锁死度:{inp['mutual_stiffness']:.2f})" if inp['mutual_has_stuck'] else "")
    return f"### 输入状态:\n{state_text}\n\n### 第一人称描述:\n"


def tokenize_function(examples, tokenizer, max_length=256):
    """批量处理版本：将 prompt + 输出拼接并 tokenize"""
    prompts = [format_prompt(inp) for inp in examples["input"]]
    outputs = examples["output"]
    
    full_texts = [p + o + tokenizer.eos_token for p, o in zip(prompts, outputs)]
    model_inputs = tokenizer(
        full_texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors=None
    )
    
    labels = []
    for i, (p, o) in enumerate(zip(prompts, outputs)):
        p_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
        o_ids = tokenizer(o + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        label = [-100] * len(p_ids) + o_ids
        if len(label) > max_length:
            label = label[:max_length]
        else:
            label = label + [-100] * (max_length - len(label))
        labels.append(label)
    
    model_inputs["labels"] = labels
    return model_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/expression_dataset_v2.json")
    parser.add_argument("--model_path", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", default="models/expression_renderer_lora_v2")
    parser.add_argument("--resume_lora", type=str, default=None, help="已有 LoRA 适配器路径，用于增量训练")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    args = parser.parse_args()

    # 加载数据
    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset = Dataset.from_list(data)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        use_4bit=args.use_4bit,
        resume_lora=args.resume_lora
    )
    
    # Tokenize 数据集
    preprocess = partial(tokenize_function, tokenizer=tokenizer, max_length=args.max_length)
    train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(preprocess, batched=True, remove_columns=eval_dataset.column_names)

    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        learning_rate=args.learning_rate,
        fp16=True,
        gradient_accumulation_steps=2,
        warmup_ratio=0.05,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"模型已保存至 {args.output_dir}")


def load_model_and_tokenizer(model_path, use_4bit=True, resume_lora=None):
    """加载基础模型，并可选地加载已有 LoRA 适配器"""
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    if resume_lora and os.path.exists(resume_lora):
        # 加载已有的 LoRA 适配器（增量训练）
        model = PeftModel.from_pretrained(base_model, resume_lora, is_trainable=True)
        print(f"已加载 LoRA 适配器: {resume_lora}")
    else:
        # 首次训练，创建新的 LoRA 配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        model = get_peft_model(base_model, lora_config)
        print("创建全新 LoRA 适配器")
    
    model.print_trainable_parameters()
    return model, tokenizer


if __name__ == "__main__":
    main()