#!/usr/bin/env python3
"""测试触觉-意象库联动增强功能"""

import sys
sys.path.append('.')
from engine import ExistenceEngine


def main():
    print("初始化引擎...")
    engine = ExistenceEngine(vocab_size=10000, use_llm=False)
    print("引擎初始化成功！")
    
    print("\n1. 注入高柔软度触觉输入...")
    # 注入高柔软度触觉输入
    engine.apply_tactile(softness=0.9, temperature=0.6)
    
    print("\n2. 进行几轮对话，积累带有触觉印记的快照...")
    # 进行几轮对话，积累带有触觉印记的快照
    for i in range(3):
        response = engine.step(f"这是第 {i+1} 轮对话，测试高柔软度触觉输入")
        print(f"  回应: {response['generated_text'][:50]}...")
    
    print("\n3. 注入低柔软度触觉输入...")
    # 注入低柔软度触觉输入
    engine.apply_tactile(softness=0.2, temperature=0.4)
    
    print("\n4. 进行几轮对话，积累低柔软度触觉印记的快照...")
    # 进行几轮对话，积累低柔软度触觉印记的快照
    for i in range(2):
        response = engine.step(f"这是第 {i+4} 轮对话，测试低柔软度触觉输入")
        print(f"  回应: {response['generated_text'][:50]}...")
    
    print("\n5. 模拟高僵化状态...")
    # 模拟高僵化状态，通过连续输入相同内容
    for i in range(20):  # 增加到20次，确保僵化度上升
        response = engine.step("这是一个重复的输入，用来增加僵化度")
        # 每5次输入后检查僵化度
        if (i+1) % 5 == 0:
            stats = engine.get_statistics()
            print(f"  第 {i+1} 次输入后，僵化度: {stats['process_meta']['coupling_stiffness']:.3f}")
    
    print("\n6. 检查当前状态...")
    stats = engine.get_statistics()
    print(f"  当前僵化度: {stats['process_meta']['coupling_stiffness']:.3f}")
    print(f"  主导欲望: {stats['desire_spectrum']['dominant_desire']}")
    
    print("\n7. 测试沉思通路检索...")
    # 获取当前坐标和呼吸
    current_coord = engine.structural_coordinator.get_current_coordinate()
    current_breath = {
        'proj_intensity': engine.process_meta.get_recent_proj_intensity(),
        'nour_success': engine.process_meta.get_recent_nour_success(),
        'stiffness': engine.process_meta.get_coupling_stiffness()
    }
    
    # 执行沉思通路检索
    results = engine.dual_memory.contemplative_retrieval(current_coord, current_breath, top_k=5)
    print(f"  检索到 {len(results)} 个共鸣快照")
    for i, (snapshot, resonance) in enumerate(results):
        print(f"  快照 {i+1}: 共鸣度={resonance:.3f}, 摘要={snapshot.summary[:50]}...")
        print(f"    呼吸印记: {snapshot.breath}")
    
    print("\n8. 测试灵感火花...")
    # 测试灵感火花
    inspiration = engine.dual_memory.get_inspiration(current_coord, current_breath)
    print(f"  灵感火花: {inspiration}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
