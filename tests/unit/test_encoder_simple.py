#!/usr/bin/env python3
"""
简单测试SentenceTransformer编码能力
"""

import os
from sentence_transformers import SentenceTransformer
import numpy as np

# 获取项目根目录
project_root = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(project_root, 'models', 'all-MiniLM-L6-v2')

print(f"加载模型: {model_path}")
print(f"模型路径是否存在: {os.path.exists(model_path)}")

# 加载模型
encoder = SentenceTransformer(model_path, device='cpu')
print("模型加载成功！")

# 测试不同文本的编码
test_texts = ["苹果", "你好", "水果", "香蕉"]

for text in test_texts:
    # 编码文本
    embedding = encoder.encode([text])[0]
    print(f"\n文本: {text}")
    print(f"向量前10个元素: {embedding[:10]}")
    print(f"向量范数: {np.linalg.norm(embedding)}")

# 计算相似度
print("\n相似度矩阵:")
embs = encoder.encode(test_texts)
for i, text1 in enumerate(test_texts):
    for j, text2 in enumerate(test_texts):
        sim = np.dot(embs[i], embs[j]) / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]))
        print(f"{text1} vs {text2}: {sim:.4f}")
