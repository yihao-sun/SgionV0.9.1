#!/usr/bin/env python3
"""
训练意图分类器（DistilBERT 多分类）
"""

import json
import torch
import argparse
import os
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

INTENT_TO_ID = {
    "STATE_INQUIRY": 0,
    "EMOTION_EXPRESSION": 1,
    "VALUE_JUDGMENT": 2,
    "WALK_REQUEST": 3,
    "EMPTINESS_RESPONSE": 4,
    "GENERAL_CHAT": 5,
    "KNOWLEDGE_QUERY": 6   # 新增
}
ID_TO_INTENT = {v: k for k, v in INTENT_TO_ID.items()}

def tokenize_function(examples, tokenizer, max_length=128):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/intent_dataset.json")
    parser.add_argument("--model_name", default="./models/distilbert-base-multilingual-cased")
    parser.add_argument("--output_dir", default="models/intent_classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['text'] for item in data]
    labels = [INTENT_TO_ID[item['intent']] for item in data]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(INTENT_TO_ID), ignore_mismatched_sizes=True
    )

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
         output_dir=args.output_dir,
         num_train_epochs=args.epochs,
         per_device_train_batch_size=args.batch_size,
         per_device_eval_batch_size=args.batch_size,
         eval_strategy="epoch",
         save_strategy="epoch",
         logging_dir=os.path.join(args.output_dir, "logs"),
         logging_steps=20,
         load_best_model_at_end=True,
         metric_for_best_model="eval_loss",
     )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # 保存标签映射
    with open(os.path.join(args.output_dir, "intent_mapping.json"), 'w') as f:
        json.dump({"intent_to_id": INTENT_TO_ID, "id_to_intent": ID_TO_INTENT}, f)
    print(f"模型已保存至 {args.output_dir}")

if __name__ == "__main__":
    main()
