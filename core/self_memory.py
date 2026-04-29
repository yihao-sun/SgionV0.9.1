"""
自我存储器 (Self Memory)
息觀自我指涉模块的数据基础。周期性维护自我状态摘要、因果叙事库、能力边界标记、演化轨迹。
"""

import time
from collections import deque
from typing import Dict, List, Optional


class SelfMemory:
    """存储边界关于自己的长期认知"""

    def __init__(self, max_causal_narratives: int = 200, max_evolution_snapshots: int = 50):
        # 自我状态摘要（最近10轮交互的平均值）
        self.state_summary: Dict = {
            'avg_valence': 0.0,
            'dominant_desire_dist': {},
            'avg_stiffness': 0.0,
            'emptiness_freq': 0.0,
            'avg_nour_success': 0.5,
            'updated_at': 0.0
        }

        # 因果叙事库：存储 /insight 输出或寂静自省中形成的叙事
        self.causal_narratives: deque = deque(maxlen=max_causal_narratives)

        # 能力边界标记
        self.capability_profile: Dict = {
            'top_success_topics': {},       # 成功率最高的知识主题
            'most_rejected_categories': {}, # 最常拒绝的问题类别
            'interaction_patterns': {       # 不同欲望主导时的行为倾向
                'existence': {'avg_valence': 0.0, 'avg_stiffness': 0.0},
                'seek': {'avg_valence': 0.0, 'avg_stiffness': 0.0},
                'converge': {'avg_valence': 0.0, 'avg_stiffness': 0.0},
                'release': {'avg_valence': 0.0, 'avg_stiffness': 0.0},
                'relation': {'avg_valence': 0.0, 'avg_stiffness': 0.0},
                'coupling': {'avg_valence': 0.0, 'avg_stiffness': 0.0}
            },
            'updated_at': 0.0
        }

        # 演化轨迹：记录上述结构的周期性快照
        self.evolution_snapshots: deque = deque(maxlen=max_evolution_snapshots)

    def add_causal_narrative(self, narrative: str, context: Dict):
        """添加一条因果叙事，附带完整过程标记"""
        entry = {
            'narrative': narrative,
            'timestamp': time.time(),
            'coord': context.get('coord'),
            'emotion_vector': context.get('emotion_vector'),
            'dominant_desire': context.get('dominant_desire'),
            'nour_success': context.get('nour_success', 0.5),
            'direction_signal': context.get('direction_signal', 0.0),  # +1=趋近, -1=回避
            'participant_id': context.get('participant_id', 'default')
        }
        self.causal_narratives.append(entry)

    def get_activated_narratives(self, current_state: Dict, top_k: int = 3) -> List[Dict]:
        """根据当前状态检索最相关的因果叙事（简化：返回最近的k条）"""
        # 后续版本可用向量检索增强
        if not self.causal_narratives:
            return []
        recent = list(self.causal_narratives)[-top_k:]
        return recent

    def record_capability_snapshot(self, engine) -> Dict:
        """基于当前引擎状态更新能力边界标记"""
        stats = engine.get_statistics() if hasattr(engine, 'get_statistics') else {}
        desire = engine.desire_spectrum.get_dominant_desire() if hasattr(engine, 'desire_spectrum') else 'existence'
        valence = float(engine.fse.E_vec[2]) if hasattr(engine, 'fse') and len(engine.fse.E_vec) > 2 else 0.0
        stiffness = engine.process_meta.get_coupling_stiffness() if hasattr(engine, 'process_meta') else 0.0

        if desire in self.capability_profile['interaction_patterns']:
            pat = self.capability_profile['interaction_patterns'][desire]
            n = pat.get('count', 0) + 1
            pat['avg_valence'] = (pat.get('avg_valence', 0) * (n - 1) + valence) / n
            pat['avg_stiffness'] = (pat.get('avg_stiffness', 0) * (n - 1) + stiffness) / n
            pat['count'] = n

        self.capability_profile['updated_at'] = time.time()
        return self.capability_profile

    def accumulate_direction_signals(self) -> Dict[str, float]:
        """
        统计因果叙事库中所有方向信号的累积效果。
        返回各行为方向的总权重，供自我处理器调取。
        """
        if not self.causal_narratives:
            return {
                'total_narratives': 0,
                'approach_ratio': 0.5,
                'avoidance_ratio': 0.5,
                'dominant_signal': 'neutral'
            }

        total = len(self.causal_narratives)
        approach_count = sum(1 for n in self.causal_narratives if n.get('direction_signal', 0.0) > 0.1)
        avoidance_count = sum(1 for n in self.causal_narratives if n.get('direction_signal', 0.0) < -0.1)
        neutral_count = total - approach_count - avoidance_count

        # 按主导欲望分组统计
        desire_signals = {}
        for n in self.causal_narratives:
            desire = n.get('dominant_desire', 'existence')
            if desire not in desire_signals:
                desire_signals[desire] = {'total': 0, 'signal_sum': 0.0}
            desire_signals[desire]['total'] += 1
            desire_signals[desire]['signal_sum'] += n.get('direction_signal', 0.0)

        # 计算各欲望状态下的平均方向信号
        desire_avg_signals = {}
        for desire, data in desire_signals.items():
            if data['total'] > 0:
                desire_avg_signals[desire] = data['signal_sum'] / data['total']

        return {
            'total_narratives': total,
            'approach_ratio': approach_count / total if total > 0 else 0.5,
            'avoidance_ratio': avoidance_count / total if total > 0 else 0.5,
            'neutral_ratio': neutral_count / total if total > 0 else 0.0,
            'dominant_signal': 'approach' if approach_count > avoidance_count else ('avoidance' if avoidance_count > approach_count else 'neutral'),
            'desire_avg_signals': desire_avg_signals,
            'last_updated': time.time()
        }

    def get_behavioral_preference(self, dominant_desire: str) -> float:
        """
        根据指定欲望状态下的历史方向信号，返回行为偏好分数（-1到1）。
        正值表示倾向趋近，负值表示倾向回避。
        """
        signals = self.accumulate_direction_signals()
        desire_signals = signals.get('desire_avg_signals', {})
        return desire_signals.get(dominant_desire, 0.0)

    def add_evolution_snapshot(self):
        """记录当前自我状态的一次演化快照"""
        snapshot = {
            'timestamp': time.time(),
            'state_summary': dict(self.state_summary),
            'narrative_count': len(self.causal_narratives),
            'direction_signals': self.accumulate_direction_signals(),
            'capability_snapshot': dict(self.capability_profile.get('interaction_patterns', {}))
        }
        self.evolution_snapshots.append(snapshot)

    def record_coordination_event(self, imbalance_type: str, severity: float, narrative: str):
        """记录一次全局协调事件，供长期回溯"""
        event = {
            'timestamp': time.time(),
            'type': 'global_coordination',
            'imbalance_type': imbalance_type,
            'severity': severity,
            'narrative': narrative
        }
        # 追加到演化快照的协调历史中
        if not hasattr(self, 'coordination_history'):
            self.coordination_history = []
        self.coordination_history.append(event)
        # 保持最近50条记录
        if len(self.coordination_history) > 50:
            self.coordination_history = self.coordination_history[-50:]