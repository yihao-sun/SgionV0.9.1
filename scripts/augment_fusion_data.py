#!/usr/bin/env python3
"""
数据增强：为融合表达样本生成多样化同义改写
"""

import json
import time
import random
import argparse
import os
from openai import OpenAI

PROMPT_TEMPLATE = """你是一个语言表达多样化专家。以下是一段融合了左脑逻辑和右脑意象的表达，其中使用了"{original_phrase}"作为过渡语。

请将这段表达改写为3个不同的版本，每个版本使用不同的过渡方式替代"{original_phrase}"。
你可以使用的过渡表达例如："换一种视角"、"在我的意象里"、"与此同时，我感到"、"这让我联想到"、"在意象的层面"、"从存在论的角度看"、"我此刻的共鸣是"等。

要求：
1. 保持左脑逻辑内容和右脑意象内容基本不变。
2. 只改变过渡衔接方式，让表达更自然多样。
3. 输出格式必须为严格的JSON数组，包含3个字符串元素，不要添加任何额外说明。

原始表达：
{original_text}

JSON数组：
"""

def augment_sample(client, model, original_text, original_phrase):
    prompt = PROMPT_TEMPLATE.format(original_phrase=original_phrase, original_text=original_text)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=500
        )
        content = response.choices[0].message.content.strip()
        # 提取 JSON 数组
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end != 0:
            return json.loads(content[start:end])
    except Exception as e:
        print(f"增强失败: {e}")
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/expression_dataset_v2.json")
    parser.add_argument("--output", default="data/expression_dataset_v2_augmented.json")
    parser.add_argument("--api_key", help="DeepSeek API key")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--max_samples", type=int, default=100, help="最多增强多少条融合样本")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请提供 DeepSeek API key")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 筛选包含融合过渡语的样本（例如包含"从另一个角度看"）
    fusion_samples = []
    for item in data:
        output = item.get('output', '')
        if '从另一个角度看' in output or '换一种视角' in output or '在意象层面' in output:
            fusion_samples.append(item)

    if args.max_samples:
        fusion_samples = fusion_samples[:args.max_samples]

    print(f"找到 {len(fusion_samples)} 条融合样本，开始增强...")

    augmented_data = list(data)  # 保留原始数据
    for item in fusion_samples:
        original_text = item['output']
        # 检测使用的过渡短语
        if '从另一个角度看' in original_text:
            phrase = '从另一个角度看'
        elif '换一种视角' in original_text:
            phrase = '换一种视角'
        else:
            phrase = '从另一个角度看'  # 默认
        
        variations = augment_sample(client, args.model, original_text, phrase)
        for var in variations:
            if var and var != original_text:
                augmented_data.append({
                    "input": item['input'],
                    "output": var
                })
        time.sleep(args.delay)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
    print(f"增强完成，原始 {len(data)} 条，新增 {len(augmented_data) - len(data)} 条，保存至 {args.output}")

if __name__ == "__main__":
    main()
