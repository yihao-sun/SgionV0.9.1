#!/usr/bin/env python3
"""测试触觉-内感受闭环"""

import sys
sys.path.append('.')
from engine import ExistenceEngine

def main():
    engine = ExistenceEngine(vocab_size=10000, use_llm=False)
    print("初始状态:")
    print(f"  V_phys: {engine.bi.V_phys:.3f}")
    print(f"  stiffness: {engine.process_meta.get_coupling_stiffness():.3f}")
    print(f"  触觉状态: {engine.bi.get_tactile_stats()}")
    
    # 模拟柔软触觉输入
    print("\n施加柔软触觉 (softness=0.9)...")
    engine.apply_tactile(softness=0.9, temperature=0.6)
    engine.bi.update()
    print(f"  V_phys: {engine.bi.V_phys:.3f}")
    print(f"  stiffness: {engine.process_meta.get_coupling_stiffness():.3f}")
    print(f"  触觉状态: {engine.bi.get_tactile_stats()}")
    
    # 模拟粗粝触觉输入
    print("\n施加粗粝触觉 (softness=0.2)...")
    engine.apply_tactile(softness=0.2, temperature=0.4)
    engine.bi.update()
    print(f"  V_phys: {engine.bi.V_phys:.3f}")
    print(f"  stiffness: {engine.process_meta.get_coupling_stiffness():.3f}")
    print(f"  触觉状态: {engine.bi.get_tactile_stats()}")

if __name__ == "__main__":
    main()
