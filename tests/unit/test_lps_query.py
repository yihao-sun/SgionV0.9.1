#!/usr/bin/env python3
"""
测试LPS查询准确性
"""

import sys
import os
import pickle

# 添加上级目录到 Python 搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from core.lps import LPS


def main():
    """
    主函数
    """
    import os
    # 加载之前保存的LPS种子数据
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, "data")
    lps_seed_path = os.path.join(data_dir, "lps_seed.pkl")
    
    if os.path.exists(lps_seed_path):
        print(f"加载 LPS 种子数据: {lps_seed_path}")
        with open(lps_seed_path, "rb") as f:
            lps = pickle.load(f)
        print(f"LPS 中包含 {len(lps.metadata)} 个可能性")
        
        # 重新初始化encoder以确保正常工作 - 使用多语言模型
        model_path = os.path.join(project_root, 'models', 'paraphrase-multilingual-MiniLM-L12-v2')
        print(f"重新加载多语言模型: {model_path}")
        # 禁用Hugging Face连接
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        from sentence_transformers import SentenceTransformer
        lps.encoder = SentenceTransformer(model_path, device='cpu')
        print("多语言模型重新加载成功！")
    else:
        print(f"未找到 LPS 种子数据: {lps_seed_path}")
        # 创建新的LPS实例
        lps = LPS()
        
        # 读取种子文本
        seed_file = "data/seed_texts.txt"
        if os.path.exists(seed_file):
            print(f"从 {seed_file} 读取种子文本...")
            with open(seed_file, 'r', encoding='utf-8') as f:
                seed_texts = [line.strip() for line in f if line.strip()]
            
            # 添加种子文本到LPS
            print(f"添加 {len(seed_texts)} 条种子文本到 LPS...")
            for text in seed_texts:
                lps.add(text)
            
            # 保存LPS状态
            os.makedirs(data_dir, exist_ok=True)
            with open(lps_seed_path, "wb") as f:
                pickle.dump(lps, f)
            print(f"种子数据已保存到: {lps_seed_path}")
        else:
            print("未找到 seed_texts.txt")
            return
    
    # 测试查询
    test_inputs = ["苹果", "水果", "你好", "香蕉"]
    
    for inp in test_inputs:
        print(f"\n查询: {inp}")
        # 生成查询向量
        query_vec = lps.encoder.encode([inp])[0]
        # 打印查询向量的前10个元素
        print(f"  查询向量前10个元素: {query_vec[:10]}")
        # 查询LPS
        results = lps.query(query_vec, k=5, min_potency=-float('inf'))
        # 打印结果
        for i, result in enumerate(results):
            print(f"  Top {i+1}: {result['text']} (distance={result['distance']:.3f})")
            # 打印结果向量的前10个元素
            if hasattr(result, 'embedding') or 'embedding' in result:
                embedding = result.get('embedding', None)
                if embedding is not None:
                    print(f"    嵌入向量前10个元素: {embedding[:10]}")


if __name__ == "__main__":
    main()