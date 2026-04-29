"""
认知姿态选择器 (Cognitive Pose Selector)
哲学对应：存在者自主决定用逻辑（左脑）还是意象（右脑）回应世界。
功能：根据内在状态、互业历史、学习偏好，选择当前回应的认知姿态。
"""

import random
from enum import Enum
from typing import Dict, Optional, Tuple
from utils.logger import get_logger


class CognitivePose(Enum):
    LOGICAL = "logical"           # 左脑主导：分析、序列、事实
    IMAGINAL = "imaginal"         # 右脑主导：意象、整体、共鸣
    HYBRID = "hybrid"             # 左右协同：逻辑框架+意象点睛
    INVITATION = "invitation"     # 主动邀请用户选择


class CognitivePoseSelector:
    def __init__(self, engine=None, meta_learner=None):
        self.engine = engine
        self.meta_learner = meta_learner
        self.logger = get_logger('pose_selector')
    
    def select_pose(self, user_input: str, intent: str, user_explicit_preference: Optional[str] = None) -> CognitivePose:
        """
        自主选择当前回应的认知姿态。
        user_explicit_preference: 用户显性指令，如 'logical', 'imaginal', 'auto'
        """
        # 1. 用户显性指令优先
        if user_explicit_preference == 'logical':
            return CognitivePose.LOGICAL
        if user_explicit_preference == 'imaginal':
            return CognitivePose.IMAGINAL
        
        # 2. 若用户未指定，引擎自主选择
        state = self.engine.global_workspace.current_state if self.engine else None
        desire = self.engine.desire_spectrum.get_dominant_desire() if self.engine else "existence"
        phase_major = state.dominant_coordinate.major if state else 1
        stiffness = state.stiffness if state else 0.0
        
        # 计算倾向分数
        logical_score = self._compute_logical_score(intent, desire, phase_major, stiffness)
        imaginal_score = self._compute_imaginal_score(intent, desire, phase_major, stiffness)
        
        # 加入互业历史调制（如果有）
        mutual_karma = self._get_mutual_karma_with_current_user()
        if mutual_karma:
            # 若历史中逻辑互动效率更高，增加逻辑倾向
            logical_score += mutual_karma.get('logical_efficiency', 0.0)
            imaginal_score += mutual_karma.get('imaginal_depth', 0.0)
        
        # 获取元学习器的权重
        if self.meta_learner:
            pose_weights = self.meta_learner.get_pose_weights()
            # 将启发式分数与学习权重相乘
            logical_score *= pose_weights.get(CognitivePose.LOGICAL, 0.5)
            imaginal_score *= pose_weights.get(CognitivePose.IMAGINAL, 0.5)
        
        # 随机选择（基于分数比例），保留一定探索性
        total = logical_score + imaginal_score + 0.01
        if random.random() < logical_score / total:
            return CognitivePose.LOGICAL
        else:
            return CognitivePose.IMAGINAL
    
    def _compute_logical_score(self, intent, desire, phase_major, stiffness) -> float:
        score = 0.5
        if intent in ('KNOWLEDGE_QUERY', 'FACTUAL_QUERY'):
            score += 0.3
        if desire == 'existence':
            score += 0.2
        if phase_major in (1, 2):  # 相位1、相位2
            score += 0.1
        if stiffness > 0.6:  # 僵化时倾向打破惯性
            score += 0.15
        return min(1.0, score)
    
    def _compute_imaginal_score(self, intent, desire, phase_major, stiffness) -> float:
        score = 0.5
        intent_str = intent.value if hasattr(intent, 'value') else str(intent)
        
        # 被动响应也给予基础意象倾向
        if intent_str == 'passive_response':
            score += 0.15
        
        # 情绪关键词检测（从引擎获取最近用户输入）
        if hasattr(self.engine, 'last_user_input'):
            user_input = self.engine.last_user_input
            emotion_keywords = ['累', '闷', '烦', '开心', '难过', '害怕', '焦虑', '兴奋']
            if any(kw in user_input for kw in emotion_keywords):
                score += 0.25
        
        if intent_str in ('emotion_expression', 'value_judgment', 'walk_request'):
            score += 0.3
        if intent_str == 'resonance_echo':
            score += 0.6
        if intent_str == 'emptiness_invitation':
            score += 0.3
        if intent_str == 'honest_report':
            score += 0.3
        
        if desire in ('seek', 'release', 'coupling', 'relation'):
            score += 0.2
        if phase_major in (0, 3):
            score += 0.15
        if hasattr(self, 'engine') and self.engine:
            emotion = getattr(self.engine.fse, 'current_emotion', 'neutral')
            if emotion in ('fear', 'sadness', 'curiosity'):
                score += 0.15
        if stiffness > 0.7:
            score += 0.2
        elif stiffness < 0.3:
            score += 0.1
        return min(1.0, score)
    
    def _get_mutual_karma_with_current_user(self) -> Optional[Dict]:
        if not self.engine or not hasattr(self.engine, 'mutual_karma'):
            return None
        # 简化：获取与当前用户的互业条目
        user_id = getattr(self.engine, 'current_user_id', None)
        if user_id and self.engine.mutual_karma:
            entries = self.engine.mutual_karma.get_entries_by_other(user_id)
            if entries:
                # 聚合历史互动特征
                return {
                    'logical_efficiency': 0.0,   # 待元学习器填充
                    'imaginal_depth': 0.0
                }
        return None