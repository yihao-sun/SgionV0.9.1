#!/usr/bin/env python3
"""批量学习种子文本，每条自动调用 /learn 流程"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import ExistenceEngine
from core.knowledge_source import TextSource

engine = ExistenceEngine(vocab_size=10000, use_llm=True)

# 要学习的文件列表
files = ['data/seed_texts/common_sense.txt', 'data/seed_texts/daily_life.txt', 'data/seed_texts/existence_philosophy.txt']

for filepath in files:
    if not os.path.exists(filepath):
        print(f"文件不存在，跳过：{filepath}")
        continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按段落分割，过滤太短的片段
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 20]
    print(f"\n文件 {filepath}：共 {len(paragraphs)} 段")
    
    for i, para in enumerate(paragraphs):
        source = TextSource(para, identifier=f"{filepath}#{i}")
        print(f"  [{i+1}/{len(paragraphs)}] 学习: {para[:50]}...")
        try:
            result = engine.document_learner.learn(source)
            print(f"    分块={result['chunks_processed']}, 三元组={result['triplets_extracted']}, 新知识={result['new_knowledge_count']}")
        except Exception as e:
            print(f"    错误: {e}")
        time.sleep(0.5)  # 避免过快调用

engine.lps.save(os.path.join('data', 'lps_seed'))
print("\n全部完成。")
