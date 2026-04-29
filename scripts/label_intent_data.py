#!/usr/bin/env python3
"""
使用 DeepSeek API 为对话文本标注意图类别。
意图类别：STATE_INQUIRY, EMOTION_EXPRESSION, VALUE_JUDGMENT, WALK_REQUEST, EMPTINESS_RESPONSE, GENERAL_CHAT
"""

import json
import time
import random
import argparse
import os
from openai import OpenAI
from tqdm import tqdm

INTENT_TYPES = [
    "STATE_INQUIRY",       # 询问引擎状态
    "EMOTION_EXPRESSION",  # 表达情绪或困惑
    "VALUE_JUDGMENT",      # 寻求价值判断或选择建议
    "WALK_REQUEST",        # 请求共同漫步
    "EMPTINESS_RESPONSE",  # 对空性邀请的回应
    "GENERAL_CHAT",        # 普通对话
    "KNOWLEDGE_QUERY"      # 事实问答、知识查询
]

PROMPT_TEMPLATE = """你是一个对话意图分析专家。请分析以下用户输入，判断其意图类别。

意图类别定义：
- STATE_INQUIRY：用户询问引擎自身状态，如"你状态如何"、"你感觉怎么样"、"/state"
- EMOTION_EXPRESSION：用户表达自身情绪或困惑，如"我今天很闷"、"不知道为什么开心不起来"
- VALUE_JUDGMENT：用户寻求价值判断或选择建议，如"该不该出门"、"选A还是B"、"这样做对吗"
- WALK_REQUEST：用户请求共同漫步，如"我们走走"、"一起漫步"、"带我走走"
- EMPTINESS_RESPONSE：用户对引擎发起的空性邀请做出回应，如"好"、"试试"、"嗯"
- GENERAL_CHAT：普通日常对话，不属于以上任何类别
- KNOWLEDGE_QUERY：事实问答、知识查询，如"地球直径是多少"、"Python如何实现装饰器"

请输出 JSON 格式：{{"intent": "类别", "confidence": 0.0-1.0}}

用户输入：
{text}
"""

def label_text(text, client, model="deepseek-chat"):
    # 对文本中的花括号进行转义，避免 format 函数解析错误
    text_escaped = text.replace('{', '{{').replace('}', '}}')
    prompt = PROMPT_TEMPLATE.format(text=text_escaped)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )
        content = response.choices[0].message.content
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(content[start:end])
    except Exception as e:
        print(f"标注失败: {e}")
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入文本文件，每行一条")
    parser.add_argument("--output", default="data/intent_dataset.json")
    parser.add_argument("--api_key", help="DeepSeek API key")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--max_samples", type=int, default=500)
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

    dataset = []
    for text in tqdm(texts, desc="标注意图"):
        label = label_text(text, client, args.model)
        if label:
            dataset.append({"text": text, "intent": label.get("intent", "GENERAL_CHAT"), 
                           "confidence": label.get("confidence", 0.5)})
        time.sleep(args.delay)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"已标注 {len(dataset)} 条数据，保存至 {args.output}")

if __name__ == "__main__":
    main()
