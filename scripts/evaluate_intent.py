#!/usr/bin/env python3
"""评估意图分类器准确率"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import argparse

INTENT_TO_ID = {
    "STATE_INQUIRY": 0, "EMOTION_EXPRESSION": 1, "VALUE_JUDGMENT": 2,
    "WALK_REQUEST": 3, "EMPTINESS_RESPONSE": 4, "GENERAL_CHAT": 5
}
ID_TO_INTENT = {v: k for k, v in INTENT_TO_ID.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/intent_dataset.json")
    parser.add_argument("--model_path", default="models/intent_classifier")
    args = parser.parse_args()

    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['text'] for item in data]
    labels = [INTENT_TO_ID[item['intent']] for item in data]

    _, X_test, _, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)

    correct = 0
    for text, true_id in zip(X_test, y_test):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=-1).item()
        if pred_id == true_id:
            correct += 1

    acc = correct / len(y_test)
    print(f"测试集准确率: {acc:.2%} ({correct}/{len(y_test)})")

if __name__ == "__main__":
    main()
