#!/usr/bin/env python3
"""测试欲望光谱集成到引擎中"""

import sys
sys.path.append('.')
from engine import ExistenceEngine


def main():
    print("初始化引擎...")
    engine = ExistenceEngine(vocab_size=10000, use_llm=False)
    print("引擎初始化成功！")
    
    print("\n1. 初始状态测试:")
    stats = engine.get_statistics()
    desire_stats = stats['desire_spectrum']
    print(f"主导欲望: {desire_stats['dominant_desire']}")
    print("欲望强度:")
    for desire, intensity in desire_stats['intensities'].items():
        print(f"  {desire}: {intensity:.3f}")
    print("感知敏感度:")
    for sense, sensitivity in desire_stats['sensitivity'].items():
        print(f"  {sense}: {sensitivity:.3f}")
    
    print("\n2. 模拟用户输入...")
    # 模拟用户输入，触发引擎处理
    engine.step("你好，我是一个测试用户")
    
    print("\n3. 处理输入后的状态:")
    stats = engine.get_statistics()
    desire_stats = stats['desire_spectrum']
    print(f"主导欲望: {desire_stats['dominant_desire']}")
    print("欲望强度:")
    for desire, intensity in desire_stats['intensities'].items():
        print(f"  {desire}: {intensity:.3f}")
    print("感知敏感度:")
    for sense, sensitivity in desire_stats['sensitivity'].items():
        print(f"  {sense}: {sensitivity:.3f}")
    
    print("\n4. 模拟重复输入，增加僵化度...")
    # 连续输入相同内容，增加僵化度
    for i in range(5):
        engine.step("这是一个重复的输入")
    
    print("\n5. 重复输入后的状态:")
    stats = engine.get_statistics()
    desire_stats = stats['desire_spectrum']
    process_meta_stats = stats['process_meta']
    print(f"僵化度: {process_meta_stats['coupling_stiffness']:.3f}")
    print(f"主导欲望: {desire_stats['dominant_desire']}")
    print("欲望强度:")
    for desire, intensity in desire_stats['intensities'].items():
        print(f"  {desire}: {intensity:.3f}")
    
    print("\n6. 模拟新颖输入...")
    # 输入新颖内容
    engine.step("这是一个非常新颖的输入，包含很多新信息和概念")
    
    print("\n7. 新颖输入后的状态:")
    stats = engine.get_statistics()
    desire_stats = stats['desire_spectrum']
    print(f"主导欲望: {desire_stats['dominant_desire']}")
    print("欲望强度:")
    for desire, intensity in desire_stats['intensities'].items():
        print(f"  {desire}: {intensity:.3f}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
