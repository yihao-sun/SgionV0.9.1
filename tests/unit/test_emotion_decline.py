#!/usr/bin/env python3
"""
测试情绪值下降功能
"""

import sys
import os
import time

# 添加上级目录到 Python 搜索路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine import ExistenceEngine

def main():
    """
    测试情绪值下降功能
    """
    print("初始化ExistenceEngine...")
    engine = ExistenceEngine(vocab_size=10000)
    print("引擎初始化成功！")
    
    # 重复输入"你好"5次
    print("\n开始测试情绪值下降...")
    print("重复输入'你好'5次，观察情绪值变化")
    
    for i in range(5):
        print(f"\n第{i+1}次输入: 你好")
        
        # 处理输入
        response = engine.step("你好")
        
        # 检查情绪值
        if hasattr(engine.fse, 'E_vec'):
            print(f"情绪向量: {engine.fse.E_vec}")
            # 情绪向量的第三个元素是valence
            valence = engine.fse.E_vec[2]
            print(f"Valence: {valence:.4f}")
        
        if hasattr(engine.fse, 'V_emo'):
            print(f"情绪强度 (V_emo): {engine.fse.V_emo:.4f}")
        
        if hasattr(engine.fse, 'current_emotion'):
            print(f"当前情绪: {engine.fse.current_emotion}")
        
        if hasattr(engine.fse, 'L'):
            print(f"当前L值: {engine.fse.L}")
        
        if hasattr(engine.fse, 'E_pred'):
            print(f"当前E_pred: {engine.fse.E_pred:.4f}")
        
        # 等待一下，避免输出太快
        time.sleep(0.5)
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()
