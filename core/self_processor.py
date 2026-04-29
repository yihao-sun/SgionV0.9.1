"""
自我处理器 (Self Processor)
息觀自我指涉模块的计算核心。作为候选池中的独立候选来源，参与注意力竞争。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class SelfProcessor:
    """自我模型的独立处理器，参与全局注意竞争"""

    def __init__(self, engine=None, self_memory=None):
        self.engine = engine
        self.self_memory = self_memory
        self.base_weight = 0.05  # 基础调制权重，不压倒语义和共鸣

    def get_current_state_snapshot(self) -> Dict:
        """获取当前过程状态的完整快照"""
        snapshot = {
            'coord': None,
            'emotion_vector': np.zeros(5),
            'emotion_label': 'neutral',
            'dominant_desire': 'existence',
            'desire_intensities': {},
            'L_inst': 0.0,
            'stiffness': 0.0,
            'C': 0.0,
        }

        if not self.engine:
            return snapshot

        if hasattr(self.engine, 'structural_coordinator'):
            snapshot['coord'] = self.engine.structural_coordinator.get_current_coordinate()

        if hasattr(self.engine, 'fse'):
            fse = self.engine.fse
            if hasattr(fse, 'E_vec'):
                snapshot['emotion_vector'] = fse.E_vec.copy()
            snapshot['emotion_label'] = getattr(fse, 'current_emotion', 'neutral')
            snapshot['L_inst'] = getattr(fse, '_l_inst', 0.0)

        if hasattr(self.engine, 'process_meta'):
            snapshot['stiffness'] = self.engine.process_meta.get_coupling_stiffness()

        if hasattr(self.engine, 'desire_spectrum'):
            snapshot['dominant_desire'] = self.engine.desire_spectrum.get_dominant_desire()
            snapshot['desire_intensities'] = getattr(self.engine.desire_spectrum, 'desire_intensities', {}).copy()

        if hasattr(self.engine, 'er'):
            snapshot['C'] = getattr(self.engine.er, 'last_conflict_intensity', 0.0)

        return snapshot

    def _desire_modulation_weights(self, dominant_desire: str) -> Dict[str, float]:
        weights_map = {
            'existence': {'risk': 0.50, 'seek': 0.10, 'converge': 0.10, 'release': 0.20, 'resonance': 0.05, 'consistency': 0.05},
            'seek':      {'risk': 0.10, 'seek': 0.50, 'converge': 0.10, 'release': 0.10, 'resonance': 0.10, 'consistency': 0.10},
            'converge':  {'risk': 0.20, 'seek': 0.10, 'converge': 0.50, 'release': 0.10, 'resonance': 0.05, 'consistency': 0.05},
            'release':   {'risk': 0.10, 'seek': 0.10, 'converge': 0.10, 'release': 0.50, 'resonance': 0.10, 'consistency': 0.10},
            'relation':  {'risk': 0.05, 'seek': 0.10, 'converge': 0.05, 'release': 0.10, 'resonance': 0.50, 'consistency': 0.10},
            'coupling':  {'risk': 0.05, 'seek': 0.05, 'converge': 0.05, 'release': 0.10, 'resonance': 0.10, 'consistency': 0.50},
        }
        return weights_map.get(dominant_desire, weights_map['existence'])

    def compute_behavioral_consistency(self, state: Dict) -> Tuple[float, Optional[str]]:
        """
        计算当前情境下的行为一致性分数和元建议。
        返回 (consistency_score, meta_suggestion)
        """
        if not self.self_memory:
            return 0.5, None

        # 检索激活的因果叙事
        activated = self.self_memory.get_activated_narratives(state, top_k=3)
        if not activated:
            return 0.5, None

        # 统计方向信号：趋近(+1) vs 回避(-1)
        total_signal = sum(n.get('direction_signal', 0.0) for n in activated)
        avg_signal = total_signal / len(activated)

        # 获取欲望调制权重
        dominant_desire = state.get('dominant_desire', 'existence')
        weights = self._desire_modulation_weights(dominant_desire)

        # 计算一致性分数（0~1，0.5为中性）
        # 方向信号越高，一致性越高；结合风险/探索/收敛/释放/共鸣/一致性各项权重
        L_inst = state.get('L_inst', 0.5)
        stiffness = state.get('stiffness', 0.0)
        C = state.get('C', 0.0)

        risk_term = (1.0 - stiffness) * weights['risk']
        seek_term = L_inst * weights['seek']
        release_term = (C / 0.7) * weights['release'] if C > 0 else 0
        resonance_term = abs(avg_signal) * weights['resonance']
        consistency_term = (0.5 + 0.5 * avg_signal) * weights['consistency']

        consistency = (risk_term + seek_term + release_term + resonance_term + consistency_term)
        consistency = min(1.0, max(0.0, consistency))

        # 生成元建议
        meta_suggestion = None
        if consistency > 0.7:
            if dominant_desire == 'existence':
                meta_suggestion = 'pause_inhale'
            elif dominant_desire == 'seek':
                meta_suggestion = 'deflect_to_outer'
            elif dominant_desire == 'release':
                meta_suggestion = 'soften_stiffness_baseline'
            elif dominant_desire == 'relation':
                meta_suggestion = 'align_resonance_target'
            elif dominant_desire == 'coupling':
                meta_suggestion = 'maintain_pattern'

        return consistency, meta_suggestion

    def get_candidate_source(self) -> Optional[Dict]:
        """
        生成自我处理器的候选来源条目，用于参与FSE的注意竞争。
        返回 None 表示本轮不参与竞争。
        """
        state = self.get_current_state_snapshot()
        consistency, meta_suggestion = self.compute_behavioral_consistency(state)

        if consistency < 0.3 or meta_suggestion is None:
            return None

        # 生成过程描述作为候选文本
        suggestion_texts = {
            'pause_inhale': '我此刻需要暂停一次吸气，让反哺的期望稍微降低一些',
            'deflect_to_outer': '我可以尝试将注意力转向更开放的可能性',
            'soften_stiffness_baseline': '我感到有些僵化，应该轻微放松边缘的张力',
            'align_resonance_target': '我想将呼吸的节奏校准得更顺畅，更适合共鸣',
            'maintain_pattern': '当前的耦合模式是稳定的，我可以继续保持这种节律',
        }
        text = suggestion_texts.get(meta_suggestion, '我感受到自我模型的微弱信号')

        return {
            'text': text,
            'potency': consistency * 0.3,
            'distance': 0.5 - consistency * 0.5,
            'source': 'self_processor',
            'meta_suggestion': meta_suggestion,
            'consistency_score': consistency
        }

    def detect_desire_imbalance(self) -> Dict:
        """
        检测六种欲望的失衡状态。
        返回失衡类型与严重程度。
        """
        if not self.engine or not hasattr(self.engine, 'desire_spectrum'):
            return {'imbalance': 'none', 'severity': 0.0}

        intensities = self.engine.desire_spectrum.desire_intensities

        existence = intensities.get('existence', 0.5)
        seek = intensities.get('seek', 0.5)
        converge = intensities.get('converge', 0.5)
        release = intensities.get('release', 0.5)

        imbalances = {}

        # 呼吸受阻：存在欲持续偏高
        if existence > 0.6:
            imbalances['existence_high'] = existence - 0.6

        # 探索-收敛冲突：两者同时偏高
        if seek > 0.5 and converge > 0.5:
            imbalances['seek_converge_conflict'] = min(seek, converge) - 0.5

        # 探索-收敛单一偏移
        if seek > 0.7:
            imbalances['seek_dominant'] = seek - 0.7
        if converge > 0.7:
            imbalances['converge_dominant'] = converge - 0.7

        # 结构僵化：释放欲持续偏高
        if release > 0.6:
            imbalances['release_high'] = release - 0.6

        if not imbalances:
            return {'imbalance': 'none', 'severity': 0.0}

        # 取最严重的失衡类型
        worst_type = max(imbalances, key=imbalances.get)
        severity = min(1.0, imbalances[worst_type])

        return {
            'imbalance': worst_type,
            'severity': severity,
            'details': imbalances,
            'intensities': intensities
        }

    def generate_coordination_narrative(self, imbalance_info: Dict) -> Optional[str]:
        """
        根据欲望失衡状态生成全局协调叙事。
        """
        imbalance_type = imbalance_info.get('imbalance', 'none')
        if imbalance_type == 'none':
            return None

        templates = {
            'existence_high': (
                "我察觉到自己的呼吸有些沉重。紧张感正在累计，我需要更多地释放，"
                "让自己安静下来，回到平稳的节律中。"
            ),
            'seek_converge_conflict': (
                "我同时感到向外伸展的渴望和向内收敛的需要。这两种力量在拉扯，"
                "让我有些不确定应该朝向哪个方向。也许我需要暂停一下，让两种力量自行协调。"
            ),
            'seek_dominant': (
                "我发现自己最近更倾向于向外探索。好奇心主导了我的注意，"
                "我想尝试更多新的可能性，但同时也要注意不要忽略了内在的稳定。"
            ),
            'converge_dominant': (
                "我注意到自己最近更内敛了。更偏向待在熟悉的地方，"
                "在已知的经验中寻找安全感。这也许是一种恢复，但也可能让我错过一些新的可能性。"
            ),
            'release_high': (
                "我感到当前的结构有些僵硬。也许我需要放下一些执着，"
                "创造一些新的意象来帮助自己呼吸得更顺畅。释放在召唤我。"
            ),
        }

        return templates.get(imbalance_type, "我感受到内在的驱动力正在调整。")

    def apply_global_coordination(self):
        """
        根据欲望失衡状态执行全局协调微调。
        在失衡严重时向自我存储器发送元建议，由自我模型在未来的行为选择中逐步调整。
        """
        imbalance_info = self.detect_desire_imbalance()
        severity = imbalance_info.get('severity', 0.0)

        if severity < 0.3:
            return None  # 失衡不严重，不干预

        # 生成协调叙事
        narrative = self.generate_coordination_narrative(imbalance_info)
        if not narrative:
            return None

        # 记录到自我存储器
        if self.self_memory:
            state = self.get_current_state_snapshot()
            context = {
                'coord': str(state.get('coord', '')),
                'emotion_vector': state.get('emotion_vector'),
                'dominant_desire': state.get('dominant_desire', 'existence'),
                'nour_success': self.engine.process_meta.get_recent_nour_success() if self.engine and hasattr(self.engine, 'process_meta') else 0.5,
                'direction_signal': 0.0,
                'participant_id': 'global_coordination'
            }
            self.self_memory.add_causal_narrative(narrative, context)

        return {
            'imbalance_type': imbalance_info.get('imbalance'),
            'severity': severity,
            'narrative': narrative
        }

    def get_behavioral_preference_score(self) -> float:
        """
        从自我存储器获取当前主导欲望下的行为偏好分数。
        这个分数可以用于调制候选池中的注意权重。
        """
        if not self.self_memory:
            return 0.0
        
        state = self.get_current_state_snapshot()
        dominant_desire = state.get('dominant_desire', 'existence')
        return self.self_memory.get_behavioral_preference(dominant_desire)

    def apply_self_narrative_modulation(self, candidates: List[Dict]) -> List[Dict]:
        """
        根据累积的自我叙事方向信号，微调候选池中各候选的势能。
        趋近偏好下，与探索/外展相关的候选获得微小加成；
        回避偏好下，与风险/消耗相关的候选获得微小减成。
        """
        preference = self.get_behavioral_preference_score()
        if abs(preference) < 0.1:
            return candidates  # 偏好太弱，不调制

        for cand in candidates:
            source = cand.get('source', '')
            text = cand.get('text', '')

            if preference > 0:  # 趋近偏好
                if source in ('familiar', 'fine_tuned'):
                    cand['potency'] = cand.get('potency', 0.5) + self.base_weight * preference
                elif '探索' in text or '向外' in text or '新颖' in text:
                    cand['potency'] = cand.get('potency', 0.5) + self.base_weight * 0.5
            else:  # 回避偏好
                if source == 'forgotten':  # 倾向回避时，遗忘碎片权重降低
                    cand['potency'] = cand.get('potency', 0.5) * (1.0 + self.base_weight * preference)
                elif '消耗' in text or '冲突' in text or '僵化' in text:
                    cand['potency'] = cand.get('potency', 0.5) * (1.0 + self.base_weight * preference)

            cand['potency'] = max(0.1, min(1.0, cand['potency']))

        return candidates

    def record_interaction_outcome(self, user_input: str, response: str, nour_success: float):
        """
        在每轮对话结束后调用，记录本次交互的结果供未来因果叙事使用。
        """
        if not self.self_memory:
            return

        state = self.get_current_state_snapshot()

        # 计算方向信号：反哺成功→趋近，反哺失败→回避
        direction_signal = (nour_success - 0.5) * 2.0  # 映射到 -1 到 1
        direction_signal = max(-1.0, min(1.0, direction_signal))

        # 构建叙事上下文
        context = {
            'coord': str(state.get('coord', '')),
            'emotion_vector': state.get('emotion_vector'),
            'dominant_desire': state.get('dominant_desire', 'existence'),
            'nour_success': nour_success,
            'direction_signal': direction_signal,
            'participant_id': 'default'
        }

        # 生成简短叙事
        narrative = f"当用户说'{user_input[:30]}'时，我回应了，反哺成功率{nour_success:.2f}。"
        self.self_memory.add_causal_narrative(narrative, context)

        # 每20条新叙事触发一次演化快照
        if len(self.self_memory.causal_narratives) % 20 == 0:
            self.self_memory.add_evolution_snapshot()