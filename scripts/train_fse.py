#!/usr/bin/env python3
"""
FSE强化学习训练脚本

使用PPO算法训练FSE的注意力选择策略和情绪更新参数
"""

import sys
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.rl_env import FSEEnv
from fse import FantasySuperpositionEngine
from er import EmptinessRegulator
from utils.config_loader import Config


def train_fse():
    """
    训练FSE策略
    """
    # 初始化配置
    config = Config()
    
    # 初始化FSE和ER
    fse = FantasySuperpositionEngine(
        embedding_dim=512,
        **config.get('fse', {})
    )
    
    er = EmptinessRegulator(
        embedding_dim=512,
        **config.get('er', {})
    )
    
    # 创建环境
    env = FSEEnv(fse, er, max_steps=100)
    
    # 初始化PPO模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5
    )
    
    # 训练模型
    print("开始训练FSE策略...")
    model.learn(total_timesteps=100000)
    
    # 保存模型
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'fse_policy.pth')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"模型保存成功: {model_path}")


if __name__ == "__main__":
    train_fse()
