#!/usr/bin/env python3
"""
测试引擎基本功能
"""

import sys
import os

# 添加上级目录到 Python 搜索路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine import ExistenceEngine

def main():
    """
    测试引擎基本功能
    """
    print("初始化ExistenceEngine...")
    engine = ExistenceEngine(vocab_size=10000)
    print("引擎初始化成功！")
    
    # 测试查询
    test_inputs = ["苹果", "水果", "你好"]
    
    for inp in test_inputs:
        print(f"\n测试输入: {inp}")
        response = engine.step(inp)
        print(f"引擎响应: {response}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()
