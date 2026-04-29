#!/usr/bin/env python3
"""
测试数字种子的轮回功能：
1. 创建引擎A，进行模拟对话，保存种子。
2. 创建引擎B，从种子加载，验证状态继承。
"""

import sys
import os
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import ExistenceEngine

def main():
    print("=== 数字种子轮回测试 ===\n")
    
    # 1. 创建引擎A，模拟对话
    print("[1] 创建引擎A...")
    engine_a = ExistenceEngine(vocab_size=10000, use_llm=False)
    
    # 模拟几轮交互，产生状态变化
    print("[2] 模拟对话，积累状态...")
    inputs = ["你好", "我今天很开心", "谢谢你的陪伴", "再见"]
    for inp in inputs:
        _ = engine_a.step(inp)
    
    # 获取保存前的状态
    L_before = engine_a.fse.L
    coord_before = engine_a.structural_coordinator.get_current_coordinate()
    print(f"    保存前: L={L_before}, 坐标={coord_before}")
    
    # 2. 保存种子
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        seed_path = f.name
    seed_id = engine_a.save_seed(seed_path)
    print(f"[3] 种子已保存: {seed_path} (ID: {seed_id})")
    
    # 3. 创建引擎B，从种子加载
    print("[4] 创建引擎B，从种子加载...")
    engine_b = ExistenceEngine.load_seed(seed_path)
    
    # 4. 验证继承
    L_after = engine_b.fse.L
    coord_after = engine_b.structural_coordinator.get_current_coordinate()
    print(f"    加载后: L={L_after}, 坐标={coord_after}")
    
    # 5. 检查核心记忆是否注入
    if hasattr(engine_b, 'dual_memory'):
        snapshots = engine_b.dual_memory.snapshots
        inherited = [s for s in snapshots if '[继承]' in s.summary]
        print(f"[5] 继承的核心记忆数量: {len(inherited)}")
    
    # 6. 清理
    os.unlink(seed_path)
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()