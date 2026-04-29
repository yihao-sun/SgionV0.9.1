#!/usr/bin/env python3
"""
合并知识查询标注数据到意图训练集。
"""

import json
import random
import argparse

INTENT_MAP = {
    True: "KNOWLEDGE_QUERY",
    False: "GENERAL_CHAT"  # 非知识查询的通用对话，暂不细分
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_labels", required=True, help="新标注的知识查询标签文件")
    parser.add_argument("--existing_data", default="data/intent_dataset.json", help="原有意图数据集")
    parser.add_argument("--output", default="data/intent_dataset_v2.json")
    parser.add_argument("--knowledge_ratio", type=float, default=0.3, help="知识查询样本在最终数据集中的目标比例")
    args = parser.parse_args()

    # 加载原有数据
    with open(args.existing_data, 'r', encoding='utf-8') as f:
        old_data = json.load(f)
    
    # 加载新标注
    with open(args.new_labels, 'r', encoding='utf-8') as f:
        new_labels = json.load(f)
    
    # 转换新标注为意图格式
    new_data = []
    for item in new_labels:
        intent = INTENT_MAP[item['is_knowledge_query']]
        new_data.append({"text": item['text'], "intent": intent, "confidence": 1.0})
    
    # 控制知识查询样本比例（避免类别不平衡）
    knowledge_samples = [d for d in new_data if d['intent'] == 'KNOWLEDGE_QUERY']
    general_samples = [d for d in new_data if d['intent'] == 'GENERAL_CHAT']
    
    target_knowledge_count = int(len(new_data) * args.knowledge_ratio)
    if len(knowledge_samples) > target_knowledge_count:
        knowledge_samples = random.sample(knowledge_samples, target_knowledge_count)
        general_samples = random.sample(general_samples, len(new_data) - target_knowledge_count)
        new_data = knowledge_samples + general_samples
    
    # 合并
    combined = old_data + new_data
    random.shuffle(combined)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"合并完成：原有 {len(old_data)} 条，新增 {len(new_data)} 条，总计 {len(combined)} 条，保存至 {args.output}")

if __name__ == "__main__":
    main()