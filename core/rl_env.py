"""
强化学习环境 (RL Environment)
将 Existence Engine 封装为 Gymnasium 环境，用于训练注意力温度策略。
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

class FSEEnv(gym.Env):
    def __init__(self, engine, max_steps=100):
        super().__init__()
        self.engine = engine
        self.max_steps = max_steps
        self.current_step = 0
        
        # 状态向量: [E_vec(5), L/L_max, N_neg, stiffness, V_emo] -> 9维
        # 归一化后各维度均在 [0,1] 或 [-1,1] 范围内，这里统一映射到 [-1,1] 以适配策略网络
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        
        # 动作空间: 温度参数 tau，范围 [0.1, 2.0]
        self.action_space = spaces.Box(low=0.1, high=2.0, shape=(1,), dtype=np.float32)
        
        # 预设一些用于训练的语料（简化的情绪表达）
        self.train_corpus = [
            "我很开心", "我感到悲伤", "这让我生气", "我有点害怕", "我对这个很好奇",
            "今天天气不错", "我遇到了一些困难", "你能帮我吗", "我想了解更多",
            "这很有趣", "我不确定", "让我想想", "我同意", "我不同意",
            "你好", "再见", "谢谢", "对不起"
        ]
        self.corpus_index = 0
        
        self._stiffness_prev = 0.0
    
    def _get_obs(self):
        """构建9维观测向量，归一化到[-1,1]"""
        fse = self.engine.fse
        pm = self.engine.process_meta
        
        # E_vec 已经在[-1,1]范围
        e_vec = fse.E_vec.copy() if hasattr(fse, 'E_vec') else np.zeros(5)
        
        # L_inst 归一化到 [-1,1]：0 -> -1, 1 -> 1
        l_inst = getattr(fse, '_l_inst', 0.0)
        L_norm = l_inst * 2 - 1  # 0~1 -> -1~1
        L_norm = np.clip(L_norm, -1.0, 1.0)
        
        # N_neg 归一化到 [-1,1]，使用 tanh 函数进行平滑归一化
        N_neg = fse.N_neg if hasattr(fse, 'N_neg') else 0.0
        # 使用 tanh 函数将任意值映射到 [-1, 1]
        N_neg_norm = np.tanh(N_neg / 100)  # 除以 100 是为了控制 tanh 的斜率
        
        # stiffness 在 [0,1]，归一化到 [-1,1]
        stiffness = pm.get_coupling_stiffness() if pm else 0.0
        stiffness_norm = stiffness * 2 - 1
        stiffness_norm = np.clip(stiffness_norm, -1.0, 1.0)
        
        # V_emo 在 [0,1]，归一化到 [-1,1]
        V_emo = fse.V_emo if hasattr(fse, 'V_emo') else 1.0
        V_emo_norm = V_emo * 2 - 1
        V_emo_norm = np.clip(V_emo_norm, -1.0, 1.0)
        
        obs = np.concatenate([e_vec, [L_norm, N_neg_norm, stiffness_norm, V_emo_norm]])
        return obs.astype(np.float32)
    
    def _get_user_input(self):
        """循环采样训练语料"""
        text = self.train_corpus[self.corpus_index % len(self.train_corpus)]
        self.corpus_index += 1
        return text
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        # 如果有重置方法则调用，否则手动重置关键状态
        if hasattr(self.engine, 'reset'):
            self.engine.reset()
        else:
            if hasattr(self.engine.fse, 'reset'):
                self.engine.fse.reset()
            if hasattr(self.engine, 'process_meta') and self.engine.process_meta:
                self.engine.process_meta.reset_coupling()
        self._stiffness_prev = 0.0
        return self._get_obs(), {}
    
    def step(self, action):
        # 将动作 tau 存入 fse，供 step 中注意力计算使用
        tau = float(action[0])
        if hasattr(self.engine.fse, 'rl_tau'):
            self.engine.fse.rl_tau = tau
        
        # 获取用户输入并执行引擎一步
        user_input = self._get_user_input()
        _ = self.engine.step(user_input)  # 响应我们可能不关心
        
        self.current_step += 1
        
        # 获取最新状态
        obs = self._get_obs()
        
        # 计算奖励
        fse = self.engine.fse
        pm = self.engine.process_meta
        l_inst = getattr(fse, '_l_inst', 0.5)
        N_neg = fse.N_neg if hasattr(fse, 'N_neg') else 0.0
        stiffness = pm.get_coupling_stiffness() if pm else 0.0
        
        # 僵化惩罚、平衡奖励（针对 l_inst=0.5 平衡状态）、否定惩罚
        reward = -0.5 * stiffness + 0.3 * (1.0 - abs(l_inst - 0.5) / 0.5) - 0.2 * N_neg
        
        # 自发重启奖励
        if hasattr(self.engine, 'er') and self.engine.er.trigger_count > getattr(self, '_last_trigger_count', 0):
            reward += 1.0
            self._last_trigger_count = self.engine.er.trigger_count
        else:
            self._last_trigger_count = getattr(self.engine, 'er', None).trigger_count if hasattr(self.engine, 'er') else 0
        
        # 终止条件：达到最大步数
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {'tau': tau, 'reward_components': {'stiffness_penalty': -0.5*stiffness, 'balance': 0.3*(1-abs(L-7.5)/7.5), 'neg_penalty': -0.2*N_neg}}
        
        return obs, reward, terminated, truncated, info
