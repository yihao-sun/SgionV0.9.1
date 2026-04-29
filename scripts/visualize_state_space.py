#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
状态空间可视化脚本
使用 PCA 将五维情绪向量降到 2D 并绘图，展示吸引子分布和状态覆盖情况
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pickle
import os
import sys
sys.path.append('.')

from engine import ExistenceEngine

def load_state_trajectory():
    """
    加载状态轨迹数据
    从 MetaEmotionRegulator 的 state_trajectory 中获取
    """
    # 创建引擎实例
    engine = ExistenceEngine(vocab_size=10000)
    
    # 运行一些步骤来生成状态轨迹
    print("Generating state trajectory...")
    for step in range(5000):
        # 随机输入
        inputs = [
            "我同时要做三件事，忙不过来",
            "任务A快完成了，但任务B还没开始",
            "我不知道接下来会发生什么",
            "结果完全无法预测",
            "我感觉被孤立了",
            "大家都不理解我",
            "你越界了，请停止",
            "这是我的底线",
            "我很焦虑，压力很大",
            "太棒了，我很开心",
            "有什么新奇的东西吗？",
            "我的CPU占用100%，快死机了！",
            "我同时和5个人聊天，每个都要求立即回复。",
            "我已经完成了所有任务，现在无事可做。"
        ]
        
        import random
        user_input = random.choice(inputs)
        
        # 基于任务完成度的真实行动成功率
        if "忙不过来" in user_input:
            action_success = 0.2   # 多任务通常失败
        elif "快死机了" in user_input:
            action_success = 0.1
        elif "被孤立" in user_input:
            action_success = 0.3
        elif "越界" in user_input:
            action_success = 0.6   # 边界维护可能成功
        else:
            action_success = 0.7   # 正常对话成功率高
        
        # 更新身体状态
        engine.bi.update_social_signal(user_input)
        # 更新情绪向量
        engine.fse._update_emotion_vector()
        
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
    
    # 获取状态轨迹
    trajectory = engine.fse.meta_emotion.state_trajectory
    print(f"Generated {len(trajectory)} state points")
    
    return trajectory

def visualize_state_space(trajectory):
    """
    可视化状态空间
    """
    # 提取状态向量
    vectors = np.array([t[1] for t in trajectory])
    print(f"State vectors shape: {vectors.shape}")
    
    # 使用 PCA 降维到 2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)
    print(f"Reduced shape: {reduced.shape}")
    
    # 提取吸引子中心
    engine = ExistenceEngine(vocab_size=10000)
    attractor_centers = []
    attractor_labels = []
    for attr_id, attr in engine.fse.emotion_attractor.attractors.items():
        center = attr.center
        # 将吸引子中心也降维
        center_2d = pca.transform([center])[0]
        attractor_centers.append(center_2d)
        attractor_labels.append(engine.fse.emotion_attractor._get_name_from_id(attr_id))
    
    # 绘制状态空间
    plt.figure(figsize=(12, 10))
    
    # 绘制状态点，使用时间作为颜色渐变
    scatter = plt.scatter(
        reduced[:, 0], 
        reduced[:, 1], 
        c=range(len(reduced)), 
        cmap='viridis', 
        alpha=0.6, 
        s=30
    )
    
    # 绘制吸引子中心
    attractor_centers = np.array(attractor_centers)
    plt.scatter(
        attractor_centers[:, 0], 
        attractor_centers[:, 1], 
        c='red', 
        marker='X', 
        s=100, 
        label='Attractors'
    )
    
    # 添加吸引子标签
    for i, label in enumerate(attractor_labels):
        plt.annotate(
            label, 
            (attractor_centers[i, 0], attractor_centers[i, 1]),
            fontsize=10, 
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
    # 添加颜色条
    plt.colorbar(scatter, label='Time Step')
    
    # 设置标题和标签
    plt.title("State Space Coverage with Attractors", fontsize=16, fontweight='bold')
    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.2f} variance)")
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.2f} variance)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图表
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "state_space.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved state space visualization to {output_path}")
    
    # 显示图表
    plt.show()

def main():
    """
    主函数
    """
    print("=== State Space Visualization ===")
    
    # 加载状态轨迹
    trajectory = load_state_trajectory()
    
    # 可视化状态空间
    visualize_state_space(trajectory)

if __name__ == "__main__":
    main()
