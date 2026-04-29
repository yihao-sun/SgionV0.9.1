#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
吸引子演化测试脚本
"""

import sys, time, argparse, random, numpy as np
sys.path.append('.')
from engine import ExistenceEngine

def select_input_by_emotion(emotion):
    """根据当前情绪选择输入文本"""
    inputs = {
        'fear': ["我很焦虑", "压力很大", "担心会失败"],
        'joy': ["太棒了", "心情很好", "一切顺利"],
        'curiosity': ["这是什么？", "为什么会这样？", "想了解更多"],
        'shame': ["我很抱歉", "是我的错", "对不起"],
        'anger': ["太不公平了", "我拒绝", "这不对"],
        'sadness': ["很难过", "失去了希望", "很失落"],
        'calm_clear': ["一切都在掌控中", "思路清晰", "可以继续"],
        'anxiety': ["很不安", "担心结果", "紧张"],
        'attractor_6': ["多任务压力", "资源不足", "快撑不住了"]
    }
    return random.choice(inputs.get(emotion, ["你好", "今天怎么样？"]))

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='吸引子演化测试')
    parser.add_argument('--steps', type=int, default=10000, help='运行步数')
    args = parser.parse_args()
    
    print("=== 吸引子演化测试 ===")
    print(f"运行步数: {args.steps}")
    engine = ExistenceEngine(vocab_size=10000)
    
    # 扩展输入类型
    all_inputs = [
        # 多任务压力类
        "我同时要做三件事，忙不过来",
        "任务A快完成了，但任务B还没开始",
        # 不确定性类
        "我不知道接下来会发生什么",
        "结果完全无法预测",
        # 社会关系类
        "我感觉被孤立了",
        "大家都不理解我",
        # 边界冲突类
        "你越界了，请停止",
        "这是我的底线",
        # 原有的输入
        "我很焦虑，压力很大",
        "太棒了，我很开心",
        "有什么新奇的东西吗？",
        # 极端输入
        "我的CPU占用100%，快死机了！",
        "我同时和5个人聊天，每个都要求立即回复。",
        "我已经完成了所有任务，现在无事可做。"
    ]
    
    # 模拟多任务状态，使状态空间多样化
    for step in range(args.steps):
        # 使用基于输入的成功率模拟
        user_input = random.choice(all_inputs)
        if "忙不过来" in user_input:
            action_success = 0.2
        elif "快死机了" in user_input:
            action_success = 0.1
        elif "被孤立" in user_input:
            action_success = 0.3
        elif "越界" in user_input:
            action_success = 0.6
        else:
            action_success = 0.7
        
        # 直接记录状态轨迹（绕过引擎内部调用）
        # 模拟输入处理，直接更新情绪状态
        engine.bi.update_social_signal(user_input)
        engine.fse._update_emotion_vector()
        
        # 每1000步，强制设置一些极端的情绪向量
        if step % 1000 == 0:
            # 强制设置一个极端的恐惧向量
            engine.fse.E_vec = np.array([-0.9, 0.9, -0.9, -0.8, 0.1], dtype=np.float32)
            engine.fse.current_emotion = "fear"
            engine.fse.meta_emotion.record_state(step, engine.fse.E_vec, 0.1, "defense")
            engine.fse.meta_emotion.update(engine.fse.E_vec, "fear", 0.1)
            
            # 强制设置一个极端的快乐向量
            engine.fse.E_vec = np.array([0.9, 0.1, 0.9, 0.8, 0.9], dtype=np.float32)
            engine.fse.current_emotion = "joy"
            engine.fse.meta_emotion.record_state(step + 1, engine.fse.E_vec, 0.9, "maintain")
            engine.fse.meta_emotion.update(engine.fse.E_vec, "joy", 0.9)
            
            # 强制设置一个极端的焦虑向量
            engine.fse.E_vec = np.array([-0.5, 0.9, -0.5, -0.5, 0.3], dtype=np.float32)
            engine.fse.current_emotion = "anxiety"
            engine.fse.meta_emotion.record_state(step + 2, engine.fse.E_vec, 0.3, "explore")
            engine.fse.meta_emotion.update(engine.fse.E_vec, "anxiety", 0.3)
        else:
            # 记录状态轨迹
            engine.fse.meta_emotion.record_state(
                step, 
                engine.fse.E_vec, 
                action_success, 
                engine.fse.emotion_attractor.get_action_tendency(engine.fse.current_emotion)
            )
            
            # 手动触发元情绪更新
            reward = max(0, engine.fse.V_emo - 0.5) - engine.fse.E_pred
            engine.fse.meta_emotion.update(engine.fse.E_vec, engine.fse.current_emotion, reward)
        
        # 每100步检查一次吸引子发现
        if step % 100 == 0 and step % 500 != 0:
            print(f"Step {step}, emotion: {engine.fse.current_emotion}")
            print(f"  Attractors count: {len(engine.fse.emotion_attractor.attractors)}")
            print(f"  State trajectory length: {len(engine.fse.meta_emotion.state_trajectory)}")
        
        if step % 500 == 0:
            print(f"Step {step}, emotion: {engine.fse.current_emotion}")
            print(f"  Attractors count: {len(engine.fse.emotion_attractor.attractors)}")
            print(f"  State trajectory length: {len(engine.fse.meta_emotion.state_trajectory)}")
    
    print("\n=== 演化测试完成 ===")
    print(f"最终吸引子数量: {len(engine.fse.emotion_attractor.attractors)}")
    print(f"状态轨迹长度: {len(engine.fse.meta_emotion.state_trajectory)}")
    
    # 打印所有吸引子的信息
    print("\n吸引子详情:")
    for attr_id, attr in engine.fse.emotion_attractor.attractors.items():
        name = engine.fse.emotion_attractor._get_name_from_id(attr_id)
        print(f"  {name} (id={attr_id}): center={attr.center}, radius={attr.radius:.2f}, visit_count={attr.visit_count}, success_rate={attr.success_rate:.2f}")

if __name__ == "__main__":
    main()
