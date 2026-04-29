#!/usr/bin/env python3

"""
性能分析脚本
用于诊断 Existence Engine 的响应时间问题
"""

import time
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import ExistenceEngine

def profile_performance():
    print("开始性能分析...")
    
    # 初始化引擎
    start_init = time.time()
    engine = ExistenceEngine(vocab_size=10000)
    init_time = time.time() - start_init
    print(f"引擎初始化时间: {init_time:.2f}秒")
    
    # 测试输入
    test_inputs = ["你好", "今天天气不错", "我很开心"]
    
    for i, inp in enumerate(test_inputs):
        print(f"\n测试输入 {i+1}: {inp}")
        start = time.time()
        # 构建输入张量
        import torch
        input_ids = torch.randint(0, engine.vocab_size, (1, 10))
        # 调用 forward 方法
        output = engine.forward(input_ids, input_text=inp)
        response = output.get('generated_text', '嗯。')
        elapsed = time.time() - start
        print(f"响应: {response}")
        print(f"响应时间: {elapsed:.2f}秒")
        print(f"情绪: {engine.fse.current_emotion}")
        # 移除对V_emo的访问，因为FantasySuperpositionEngine对象没有这个属性
    
    # 检查否定关系图大小
    if hasattr(engine.fse, 'neg_graph') and hasattr(engine.fse.neg_graph, 'dynamic') and hasattr(engine.fse.neg_graph.dynamic, 'nodes'):
        print(f"\n否定关系图节点数: {len(engine.fse.neg_graph.dynamic.nodes)}")
    
    print("\n性能分析完成！")

if __name__ == "__main__":
    profile_performance()