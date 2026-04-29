#!/usr/bin/env python3
"""
将微调数据集中的全部问答对注入 LPS，作为高势能知识种子。
运行方式：python scripts/inject_finetune_to_lps.py
"""

import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine import ExistenceEngine

def main():
    dataset_path = "data/ee_finetune_dataset.jsonl"
    if not os.path.exists(dataset_path):
        print(f"数据集文件不存在: {dataset_path}")
        print("请先运行微调数据生成脚本")
        return
    
    print("正在初始化引擎...")
    engine = ExistenceEngine(vocab_size=10000, use_llm=False)
    
    # 加载全部数据集
    pairs = []
    print(f"开始加载数据集: {dataset_path}")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    pairs.append((item['instruction'], item['output']))
                    if line_num % 1000 == 0:
                        print(f"  已加载 {line_num} 条...")
                except json.JSONDecodeError as e:
                    print(f"  第 {line_num} 行解析失败: {e}")
                    continue
        print(f"加载完成，共 {len(pairs)} 条微调数据")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return
    
    # 注入LPS
    added = 0
    for q, a in pairs:
        text = f"Q: {q}\nA: {a}"
        
        # 获取双坐标标签
        coord = engine.structural_coordinator.get_current_coordinate()
        objective_room = 0  # 默认
        if hasattr(engine, 'objective_classifier'):
            objective_room = engine.objective_classifier.classify(q, a)
        
        tags = {
            'type': 'semantic',
            'subjective_room': coord.as_tarot_code(),
            'subjective_major': coord.major,
            'objective_room': objective_room,
            'question': q,
            'answer': a[:100]
        }
        
        embedding = engine.lps.encoder.encode([text])[0]
        node_id = engine.lps.add_if_new(text, embedding, potency=0.85, tags=tags)
        if node_id:
            added += 1
        
        # 提取关键词并创建语义条目
        try:
            # 尝试使用工作记忆的关键词提取功能
            if hasattr(engine, 'working_memory') and hasattr(engine.working_memory, '_extract_keywords'):
                keywords = engine.working_memory._extract_keywords(q + " " + a)
                # 简单的语义条目创建（如果语义映射器存在）
                if hasattr(engine, 'semantic_mapper') and hasattr(engine.semantic_mapper, 'get_or_create_entry'):
                    for kw in keywords:
                        engine.semantic_mapper.get_or_create_entry(kw)
        except Exception as e:
            print(f"  关键词处理失败: {e}")
            pass
        
        if added % 500 == 0:
            print(f"  已注入 {added} 条...")
    
    # 保存LPS
    engine.lps.save("data/lps_seed")
    print(f"注入完成，共添加 {added} 条知识种子。")
    print(f"LPS当前总条目：{len(engine.lps.metadata)}")
    
    engine.shutdown()

if __name__ == '__main__':
    main()