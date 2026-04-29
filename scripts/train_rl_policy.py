#!/usr/bin/env python3
"""
RL 策略训练脚本
使用 PPO 训练一个简单的注意力温度调节策略。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rl_env import FSEEnv
from engine import ExistenceEngine
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch

def main():
    print("初始化引擎（不使用 LLM 以节省资源）...")
    # 从配置文件读取默认的 vocab_size
    from utils.config_loader import Config
    config = Config()
    vocab_size = config.get('lps', {}).get('vocab_size', 10000)
    engine = ExistenceEngine(vocab_size=vocab_size, use_llm=False)
    
    # 禁用状态保存，防止训练过程中生成大量大文件
    engine.save_interval = float('inf')  # 设置为无限大，禁用自动保存
    engine._save_model_version = lambda: None  # 禁用版本保存方法
    
    print("创建 RL 环境...")
    env = FSEEnv(engine, max_steps=200)
    check_env(env)
    
    print("开始训练 PPO 策略...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10)
    model.learn(total_timesteps=20000)
    
    print("保存模型...")
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/rl_policy")
    
    # 导出轻量 PyTorch 模型
    policy_net = model.policy.mlp_extractor.policy_net
    torch.save(policy_net.state_dict(), "checkpoints/rl_policy_net.pt")
    print("训练完成，模型已保存至 checkpoints/")
    
    env.close()

if __name__ == "__main__":
    main()
