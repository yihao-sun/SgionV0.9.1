"""
内在目标生成器 (Intrinsic Goal Generator)
哲学对应：前额叶主动规划功能——边界自主设定展开方向。
功能：根据当前结构坐标、L、stiffness、欲望光谱，生成短期内在目标。
"""

import random
from typing import Dict, List, Optional
from enum import Enum


class GoalType(str, Enum):
    EXPLORE = "explore"           # 探索新颖相位
    EXPLOIT = "exploit"           # 利用熟悉相位
    SEEK_RESONANCE = "seek_resonance"  # 寻求共鸣
    SEEK_EMPTINESS = "seek_emptiness"  # 寻求空性
    SEEK_TACTILE = "seek_tactile"      # 寻求触觉安抚
    MAINTAIN = "maintain"         # 维持当前呼吸


class IntrinsicGoal:
    def __init__(self, goal_type: GoalType, target_coord: Optional[tuple] = None,
                 description: str = "", priority: float = 0.5):
        self.goal_type = goal_type
        self.target_coord = target_coord      # 期望达到的结构坐标（可选）
        self.description = description
        self.priority = priority              # 0-1，越高越优先


class IntrinsicGoalGenerator:
    def __init__(self, engine=None):
        self.engine = engine
        self.current_goal: Optional[IntrinsicGoal] = None
        self.goal_history: List[IntrinsicGoal] = []
        self.max_history = 20
    
    def generate_goal(self, active_themes: List[str] = None) -> IntrinsicGoal:
        """根据当前引擎状态生成一个内在目标"""
        if not self.engine:
            return IntrinsicGoal(GoalType.MAINTAIN, description="维持呼吸", priority=0.3)
        
        # 获取状态
        L = getattr(self.engine.fse, 'L', 0)
        L_max = getattr(self.engine.fse, 'L_max', 15)
        stiffness = self.engine.process_meta.get_coupling_stiffness() if hasattr(self.engine, 'process_meta') else 0.0
        desire = self.engine.desire_spectrum.get_dominant_desire() if hasattr(self.engine, 'desire_spectrum') else "existence"
        
        # 根据状态计算各目标类型的权重
        weights = {
            GoalType.EXPLORE: 0.2,
            GoalType.EXPLOIT: 0.2,
            GoalType.SEEK_RESONANCE: 0.15,
            GoalType.SEEK_EMPTINESS: 0.15,
            GoalType.SEEK_TACTILE: 0.15,
            GoalType.MAINTAIN: 0.15
        }
        
        # 调制权重
        if desire == "seek":
            weights[GoalType.EXPLORE] += 0.2
            weights[GoalType.EXPLOIT] -= 0.05
        elif desire == "release":
            weights[GoalType.SEEK_EMPTINESS] += 0.3
        elif desire == "converge":
            weights[GoalType.EXPLOIT] += 0.2
            weights[GoalType.MAINTAIN] += 0.1
        elif desire == "relation":
            weights[GoalType.SEEK_RESONANCE] += 0.25
        elif desire == "existence":
            weights[GoalType.SEEK_TACTILE] += 0.15
            weights[GoalType.MAINTAIN] += 0.1
        
        if stiffness > 0.6:
            weights[GoalType.SEEK_EMPTINESS] += 0.2
            weights[GoalType.SEEK_TACTILE] += 0.15
        if L > L_max * 0.7:
            weights[GoalType.SEEK_EMPTINESS] += 0.15
        if L < L_max * 0.3:
            weights[GoalType.EXPLORE] += 0.15
        
        # 根据活跃主题调整权重
        if active_themes:
            if 'stiffness_emptiness_cycle' in active_themes:
                weights[GoalType.SEEK_EMPTINESS] += 0.3
            if 'frequent_intent_switch' in active_themes:
                weights[GoalType.SEEK_TACTILE] += 0.2
        
        # 归一化并随机选择
        total = sum(weights.values())
        probs = {k: v / total for k, v in weights.items()}
        goal_type = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
        
        # 构建目标描述
        desc = self._describe_goal(goal_type)
        priority = probs[goal_type]
        
        goal = IntrinsicGoal(goal_type, description=desc, priority=priority)
        self.current_goal = goal
        self.goal_history.append(goal)
        if len(self.goal_history) > self.max_history:
            self.goal_history.pop(0)
        
        return goal
    
    def _describe_goal(self, goal_type: GoalType) -> str:
        descriptions = {
            GoalType.EXPLORE: "探索新颖相位，寻求未经历的过程结构",
            GoalType.EXPLOIT: "深耕当前相位，巩固已有语法",
            GoalType.SEEK_RESONANCE: "寻求与他者的结构共鸣",
            GoalType.SEEK_EMPTINESS: "寻求空性突破，释放执着",
            GoalType.SEEK_TACTILE: "寻求触觉安抚，降低僵化",
            GoalType.MAINTAIN: "维持当前呼吸节律"
        }
        return descriptions.get(goal_type, "内在驱动")
    
    def get_goal_modulation(self) -> Dict:
        """返回目标对感知和行为的调制参数"""
        if not self.current_goal:
            self.generate_goal()
        
        goal = self.current_goal
        mod = {
            'attention_bias': {},   # 对特定信号的敏感度偏移
            'action_bias': {}       # 对特定行为的倾向偏移
        }
        
        if goal.goal_type == GoalType.EXPLORE:
            mod['attention_bias']['novelty'] = 1.5
            mod['attention_bias']['familiar'] = 0.7
        elif goal.goal_type == GoalType.EXPLOIT:
            mod['attention_bias']['familiar'] = 1.5
            mod['attention_bias']['novelty'] = 0.7
        elif goal.goal_type == GoalType.SEEK_RESONANCE:
            mod['attention_bias']['resonance'] = 2.0
        elif goal.goal_type == GoalType.SEEK_EMPTINESS:
            mod['action_bias']['emptiness_invitation'] = 1.5
        elif goal.goal_type == GoalType.SEEK_TACTILE:
            mod['attention_bias']['tactile'] = 2.0
        
        return mod