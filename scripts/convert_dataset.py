#!/usr/bin/env python3
"""
将现有数据集文件转换为正确的 JSONL 格式
运行方式：python scripts/convert_dataset.py
生成文件：data/ee_finetune_dataset.jsonl
"""

import json
import os
import re

def convert_dataset():
    input_path = "data/ee_finetune_dataset.jsonl"
    output_path = "data/ee_finetune_dataset.jsonl"
    
    if not os.path.exists(input_path):
        print(f"输入文件不存在: {input_path}")
        return
    
    print("正在解析数据集文件...")
    
    # 读取文件内容
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取所有对话样本
    # 匹配形如 {"instruction": "...", "output": "..."} 的模式
    pattern = r'\{"instruction": "([^"]*)", "output": "([^"]*)"\}'
    matches = re.findall(pattern, content)
    
    samples = []
    for match in matches:
        instruction, output = match
        # 清理转义字符
        instruction = instruction.replace('\\n', '\n').replace('\\"', '"')
        output = output.replace('\\n', '\n').replace('\\"', '"')
        samples.append({"instruction": instruction, "output": output})
    
    # 保存为正确的 JSONL 格式
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"数据集转换完成：{output_path}")
    print(f"总样本数：{len(samples)}")

if __name__ == "__main__":
    convert_dataset()