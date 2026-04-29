"""
元学习器 (Meta Learner)
功能：从交互反馈中在线学习，动态调整认知姿态权重与表达偏好。
"""

import json
import os
from typing import Dict
from core.cognitive_pose_selector import CognitivePose


class MetaLearner:
    def __init__(self, storage_path: str = "data/meta_learner.json"):
        self.storage_path = storage_path
        
        # 可学习参数：认知姿态权重（用于 CognitivePoseSelector）
        self.pose_weights: Dict[CognitivePose, float] = {
            CognitivePose.LOGICAL: 0.5,
            CognitivePose.IMAGINAL: 0.5,
            CognitivePose.HYBRID: 0.0,
            CognitivePose.INVITATION: 0.0
        }
        
        # 可学习参数：披露层级偏好（影响情境渲染器的默认 detail level）
        self.disclosure_bias: float = 0.5  # 0倾向简洁，1倾向详细
        
        # 可学习参数：融合风格偏好（0倾向纯逻辑，1倾向纯意象）
        self.fusion_style: float = 0.5
        
        # 学习率
        self.lr = 0.05
        
        self._load()
    
    def _load(self):
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                # 加载保存的权重
                saved_weights = data.get('pose_weights', {})
                for pose_name, weight in saved_weights.items():
                    pose = CognitivePose(pose_name)
                    self.pose_weights[pose] = weight
                self.disclosure_bias = data.get('disclosure_bias', 0.5)
                self.fusion_style = data.get('fusion_style', 0.5)
        except FileNotFoundError:
            pass
    
    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            'pose_weights': {p.value: w for p, w in self.pose_weights.items()},
            'disclosure_bias': self.disclosure_bias,
            'fusion_style': self.fusion_style
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def update(self, chosen_pose: CognitivePose, feedback: Dict) -> Dict:
        """
        根据交互反馈更新学习参数。
        feedback 应包含：
        - conversation_continued: bool
        - user_sentiment: float (-1 到 1)
        - user_reply_length: int (可选)
        """
        reward = 0.0
        
        # 奖励信号：对话继续 + 积极情绪
        if feedback.get('conversation_continued', False):
            reward += 0.5
        sentiment = feedback.get('user_sentiment', 0.0)
        if sentiment > 0.1:
            reward += 0.5
        elif sentiment < -0.1:
            reward -= 0.3
        
        # 根据奖励调整所选姿态的权重
        if reward > 0:
            self.pose_weights[chosen_pose] *= (1 + self.lr * reward)
        else:
            self.pose_weights[chosen_pose] *= (1 + self.lr * reward * 0.5)  # 惩罚力度稍弱
        
        # 归一化
        total = sum(self.pose_weights.values())
        for pose in self.pose_weights:
            self.pose_weights[pose] /= total
        
        # 调整披露偏好：若用户回应简短，倾向降低 detail；若用户回应详细，倾向提高
        user_reply_len = feedback.get('user_reply_length', 0)
        if user_reply_len > 20:
            self.disclosure_bias = min(1.0, self.disclosure_bias + 0.02)
        elif user_reply_len < 5:
            self.disclosure_bias = max(0.0, self.disclosure_bias - 0.02)
        
        # 调整融合风格偏好
        if feedback.get('fusion_used', False):
            # 获取当前使用的融合比例（假设在反馈中提供）
            current_fusion_ratio = feedback.get('current_fusion_ratio', 0.5)
            # 根据反馈调整融合风格
            if feedback.get('conversation_continued', False) and sentiment > 0.1:
                # 对话继续且情绪积极，向当前融合比例微调
                self.fusion_style += (current_fusion_ratio - self.fusion_style) * 0.1
            elif not feedback.get('conversation_continued', False) or sentiment < -0.1:
                # 对话中断或情绪消极，向相反方向微调
                self.fusion_style += (1.0 - current_fusion_ratio - self.fusion_style) * 0.1
            # 确保在有效范围内
            self.fusion_style = max(0.0, min(1.0, self.fusion_style))
        
        self._save()
        return {'reward': reward, 'pose_weights': self.pose_weights, 'disclosure_bias': self.disclosure_bias, 'fusion_style': self.fusion_style}
    
    def get_pose_weights(self) -> Dict[CognitivePose, float]:
        return self.pose_weights.copy()
    
    def get_disclosure_bias(self) -> float:
        return self.disclosure_bias
    
    def get_fusion_style(self) -> float:
        return self.fusion_style