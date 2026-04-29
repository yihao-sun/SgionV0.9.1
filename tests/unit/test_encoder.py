#!/usr/bin/env python3
"""
测试SentenceTransformer编码器
"""

import sys
import os
from sentence_transformers import SentenceTransformer

# 添加上级目录到 Python 搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))


def main():
    """
    主函数
    """
    # 获取项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    
    # 初始化嵌入模型（使用绝对路径加载模型）
    model_path = os.path.join(project_root, 'models', 'all-MiniLM-L6-v2')
    print(f"尝试加载模型路径: {model_path}")
    print(f"模型路径是否存在: {os.path.exists(model_path)}")
    
    try:
        # 强制使用本地模型，显式指定设备为cpu
        encoder = SentenceTransformer(model_path, device='cpu')
        print("模型加载成功！")
    except Exception as e:
        print(f"加载本地模型失败: {e}")
        # 尝试使用下载目录中的模型
        download_model_path = r'C:\Users\85971\Downloads\all-MiniLM-L6-v2'
        print(f"尝试加载下载目录模型: {download_model_path}")
        print(f"下载目录模型是否存在: {os.path.exists(download_model_path)}")
        try:
            encoder = SentenceTransformer(download_model_path, device='cpu')
            print("下载目录模型加载成功！")
        except Exception as e2:
            print(f"加载下载目录模型失败: {e2}")
            # 作为最后尝试，使用在线模型
            print("尝试使用在线模型: all-MiniLM-L6-v2")
            encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            print("在线模型加载成功！")
    
    # 测试编码
    test_inputs = ["苹果", "你好", "水果", "香蕉"]
    
    for inp in test_inputs:
        print(f"\n输入: {inp}")
        # 生成嵌入向量
        embedding = encoder.encode([inp])[0]
        # 打印嵌入向量的前10个元素
        print(f"  嵌入向量前10个元素: {embedding[:10]}")
        # 打印嵌入向量的范数
        print(f"  嵌入向量范数: {np.linalg.norm(embedding):.3f}")


if __name__ == "__main__":
    import numpy as np
    main()