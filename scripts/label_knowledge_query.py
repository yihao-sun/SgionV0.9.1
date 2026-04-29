#!/usr/bin/env python3
"""
使用 DeepSeek API 为 LCCC 文本标注是否为 KNOWLEDGE_QUERY。
"""

import json
import time
import random
import argparse
import os
from openai import OpenAI
from tqdm import tqdm

PROMPT_TEMPLATE = """判断以下用户输入是否属于"知识查询"类别。
知识查询的定义：用户在询问事实、定义、方法、客观信息，或期待一个基于知识的回答（而非情感倾诉、社交寒暄、自我探索）。
请仅回答"是"或"否"。

用户输入：
{text}
"""

def label_knowledge_query(text, client, model="deepseek-chat"):
    prompt = PROMPT_TEMPLATE.format(text=text)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        content = response.choices[0].message.content.strip()
        return content == "是"
    except Exception as e:
        print(f"标注失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入文本文件，每行一条")
    parser.add_argument("--output", default="data/knowledge_query_labels.json")
    parser.add_argument("--api_key", help="DeepSeek API key")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--max_samples", type=int, default=2000)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请提供 DeepSeek API key")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    with open(args.input, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    if args.max_samples:
        random.shuffle(texts)
        texts = texts[:args.max_samples]

    labels = []
    for text in tqdm(texts, desc="标注知识查询"):
        is_knowledge = label_knowledge_query(text, client, args.model)
        labels.append({"text": text, "is_knowledge_query": is_knowledge})
        time.sleep(args.delay)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"已标注 {len(labels)} 条数据，保存至 {args.output}")

if __name__ == "__main__":
    main()