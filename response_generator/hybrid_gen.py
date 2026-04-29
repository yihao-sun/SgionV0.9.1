import numpy as np
from .base import ResponseGenerator

class HybridGenerator(ResponseGenerator):
    def __init__(self, local_gen, api_gen, fallback_threshold: float = 0.3, max_fallback_per_session: int = 3):
        self.local = local_gen
        self.api = api_gen
        self.fallback_threshold = fallback_threshold
        self.fallback_count = 0
        self.max_fallback = max_fallback_per_session

    def _evaluate_response(self, response: str) -> float:
        """返回质量分数 0-1，越高越好"""
        if not response or len(response) < 5:
            return 0.0
        # 重复率检测
        words = response.split()
        unique_ratio = len(set(words)) / max(1, len(words))
        if unique_ratio < 0.3:
            return 0.1
        # 长度惩罚（过短）
        if len(response) < 20:
            return 0.5
        return 0.8

    def generate(self, user_input: str, S_t: np.ndarray, V_emo: float, L: int = 0, D_self: float = 0, C: float = 0) -> str:
        # 尝试本地生成
        local_resp = self.local.generate(user_input, S_t, V_emo, L, D_self, C)
        score = self._evaluate_response(local_resp)
        if score >= self.fallback_threshold:
            return local_resp
        # 需要 fallback
        if self.fallback_count < self.max_fallback:
            self.fallback_count += 1
            print(f"[Hybrid] Local quality low ({score:.2f}), using API (fallback #{self.fallback_count})")
            return self.api.generate(user_input, S_t, V_emo, L, D_self, C)
        else:
            print("[Hybrid] Max fallback reached, returning local response")
            return local_resp

    def reset_session(self):
        """重置会话，包括fallback计数"""
        self.fallback_count = 0
        # 重置API生成器的会话状态
        if hasattr(self.api, 'reset_session'):
            self.api.reset_session()
        # 重置本地生成器的会话状态
        if hasattr(self.local, 'reset_session'):
            self.local.reset_session()
