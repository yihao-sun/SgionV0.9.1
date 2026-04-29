#!/usr/bin/env python3
"""
测试ER触发功能
"""

import sys
import os
import time

# 添加上级目录到 Python 搜索路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine import ExistenceEngine

def main():
    """
    测试ER触发功能
    """
    print("初始化ExistenceEngine...")
    engine = ExistenceEngine(vocab_size=10000)
    print("引擎初始化成功！")
    
    # 模拟用户输入"你好"多次
    print("\n开始测试ER触发...")
    print("重复输入'你好'，直到L>10，观察ER是否触发")
    
    for i in range(20):  # 最多尝试20次
        print(f"\n第{i+1}次输入: 你好")
        
        # 处理输入
        response = engine.step("你好")
        
        # 检查L值
        if hasattr(engine.fse, 'L'):
            L = engine.fse.L
            print(f"当前L值: {L}")
            
            # 检查ER是否触发
            if hasattr(engine.er, 'last_conflict_intensity'):
                conflict_intensity = engine.er.last_conflict_intensity
                print(f"当前冲突强度: {conflict_intensity:.3f}")
                print(f"ER死亡阈值: {engine.er.death_threshold:.3f}")
            
            # 检查情绪值
            if hasattr(engine.fse, 'E_vec'):
                print(f"情绪向量: {engine.fse.E_vec}")
            if hasattr(engine.fse, 'current_emotion'):
                print(f"当前情绪: {engine.fse.current_emotion}")
            if hasattr(engine.fse, 'V_emo'):
                print(f"情绪强度: {engine.fse.V_emo:.3f}")
        
        # 等待一下，避免输出太快
        time.sleep(0.5)
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()
