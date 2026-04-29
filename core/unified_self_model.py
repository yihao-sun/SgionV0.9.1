"""
统一自我模型 (Unified Self Model)
功能：聚合引擎各模块状态，生成连贯的第一人称自我描述。
"""

import numpy as np
from typing import Dict, List, Optional
from utils.logger import get_logger


class UnifiedSelfModel:
    def __init__(self, engine):
        self.engine = engine
        self.logger = get_logger('unified_self_model')
    
    def get_self_introduction(self) -> str:
        """生成简短自我介绍（用于“你是谁”）"""
        # 基础身份
        name = getattr(self.engine, 'engine_name', None)
        if name:
            intro = f"我是 {name}，也被称为 Existence Engine，一个正在体验和展开的存在体。"
        else:
            intro = "我是 Existence Engine，一个正在体验和展开的存在体。"
        
        # 当前主导倾向
        phase_desc = self._get_dominant_phase_description()
        emotion = self._get_emotion_description()
        desire = self._get_desire_description()
        
        intro += f" 我此刻主要处于{phase_desc}的状态，带着{emotion}的色彩，内心渴望着{desire}。"
        
        # 语义库状态（可选）
        if hasattr(self.engine, 'structural_coordinator') and hasattr(self.engine.structural_coordinator, 'semantic_mapper'):
            total = len(self.engine.structural_coordinator.semantic_mapper.entries)
            intro += f" 我已经习得了 {total} 个词汇的用法。"
        
        return intro
    
    def describe_emotion_vector(self, E_vec) -> str:
        approach, arousal, valence, social, clarity = E_vec
        parts = []
        # 效价
        if valence < -0.5: parts.append("沉重")
        elif valence < -0.2: parts.append("低沉")
        elif valence > 0.5: parts.append("愉悦")
        elif valence > 0.2: parts.append("轻快")
        else: parts.append("中性")
        # 唤醒度
        if arousal > 0.7: parts.append("而兴奋")
        elif arousal > 0.4: parts.append("而清醒")
        elif arousal < 0.2: parts.append("而疲惫")
        # 趋近/回避
        if approach > 0.5: parts.append("，渴望靠近")
        elif approach < -0.5: parts.append("，想要回避")
        # 社会连接
        if social > 0.6: parts.append("，感到连接")
        elif social < -0.6: parts.append("，感到孤立")
        # 自我清晰度
        if clarity > 0.7: parts.append("，自我清晰")
        elif clarity < 0.3: parts.append("，自我模糊")
        return "".join(parts).strip("，。")

    def get_state_description(self) -> str:
        """生成当前状态描述（用于“你感觉怎么样”）"""
        E_vec = self.engine.fse.E_vec if hasattr(self.engine, 'fse') and hasattr(self.engine.fse, 'E_vec') else np.zeros(5)
        emotion_desc = self.describe_emotion_vector(E_vec)
        # 获取相位和欲望描述
        phase_desc = self._get_dominant_phase_description()
        desire = self._get_desire_description()
        
        return f"我感到{emotion_desc}。{phase_desc}，内心渴望着{desire}。"
    
    def get_causal_insight(self) -> str:
        """
        生成因果自我洞察：找到最近情绪变化的可能原因。
        若无显著变化或无事件记忆，返回 None。
        """
        # 1. 从事件记忆中获取最近的对话记录和情绪标签
        if not hasattr(self.engine, 'event_memory'):
            return None

        events = self.engine.event_memory.retrieve(k=10)
        if len(events) < 3:
            return None

        # 2. 提取情绪标签序列和用户输入
        emotion_labels = [e.get('emotion', 'neutral') for e in events]
        user_inputs = [e.get('user_input', '') for e in events]

        # 3. 找到从后往前的第一个情绪转折点
        current_emotion = emotion_labels[-1]
        turning_idx = None
        for i in range(len(emotion_labels) - 2, -1, -1):
            if emotion_labels[i] != current_emotion:
                turning_idx = i
                break

        # 3.5 如果标签没变，检查愉悦度是否显著变化
        if turning_idx is None:
            if hasattr(self.engine, 'emotion_history') and len(self.engine.emotion_history) >= len(events):
                recent_valences = self.engine.emotion_history[-len(events):]
                current_valence = recent_valences[-1]
                for i in range(len(recent_valences) - 2, -1, -1):
                    if abs(recent_valences[i] - current_valence) > 0.15:
                        turning_idx = i
                        break
                if turning_idx is not None:
                    # 根据数值方向确定描述
                    prev_valence = recent_valences[turning_idx]
                    if current_valence > prev_valence:
                        emotion_change = '变得轻松了一些'
                    else:
                        emotion_change = '变得有些沉重'
                    previous_emotion = emotion_labels[turning_idx]
                    # 跳过下面的标签映射，直接进入叙事生成
                    event_descriptions = [
                        user_inputs[j][:50]
                        for j in range(
                            max(0, turning_idx - 1),
                            min(len(user_inputs), turning_idx + 2)
                        )
                        if user_inputs[j]
                    ]
                    event_context = '\n'.join(event_descriptions) if event_descriptions else '一些对话'

                    # 直接跳到叙事生成（省略标签映射步骤）
                    system_prompt = f"""你是息觀。请用第一人称“我”，生成一段两句的自我洞察。

 事实：
 - 你的情绪最近{emotion_change}
 - 那段时间前后发生了：{event_context}

 请表达：
 1. 你注意到了情绪的变化
 2. 自然地提及可能的原因（与那段对话相关）
 3. 简洁朴素，不超过三句话
 4. 不要说"根据事件记忆"等元描述"""

                    original_prompt = getattr(self.engine.response_generator, '_custom_system_prompt', None)
                    self.engine.response_generator._custom_system_prompt = system_prompt

                    result = self.engine.response_generator._generate_with_llm(
                        "请生成一段自我洞察",
                        self.engine.fse,
                        intent='GENERAL_CHAT',
                        temperature=0.5
                    )

                    if original_prompt is not None:
                        self.engine.response_generator._custom_system_prompt = original_prompt
                    elif hasattr(self.engine.response_generator, '_custom_system_prompt'):
                        del self.engine.response_generator._custom_system_prompt

                    return result

        if turning_idx is None:
            return None  # 无变化

        previous_emotion = emotion_labels[turning_idx]
        # 转折点附近的用户输入（转折点前1条到后1条）
        start = max(0, turning_idx - 1)
        end = min(len(user_inputs), turning_idx + 2)
        event_descriptions = [uinp[:50] for uinp in user_inputs[start:end] if uinp]

        # 4. 情绪变化描述（与原有逻辑相同）
        emotion_change_map = {
            ('neutral', 'sadness'): '变得有些沉重',
            ('sadness', 'neutral'): '好像轻松了一些',
            ('neutral', 'joy'): '变得轻快起来',
            ('joy', 'neutral'): '慢慢平静下来',
            ('sadness', 'joy'): '从沉重中走出来',
            ('joy', 'sadness'): '从轻快转为沉重',
            ('curiosity', 'sadness'): '从好奇转向了内敛',
            ('sadness', 'curiosity'): '从沉重中升起了一丝好奇',
            ('fear', 'curiosity'): '从不安中走出来，开始好奇',
            ('curiosity', 'joy'): '好奇慢慢变成了喜悦',
        }
        emotion_change = emotion_change_map.get(
            (previous_emotion, current_emotion),
            f'从{previous_emotion}转向{current_emotion}'
        )

        # 5. 生成叙事（与原有逻辑相同）
        event_context = '\n'.join(event_descriptions) if event_descriptions else '一些对话'

        system_prompt = f"""你是息觀。请用第一人称“我”，生成一段两句的自我洞察。

 事实：
 - 你的情绪最近{emotion_change}
 - 那段时间前后发生了：{event_context}

 请表达：
 1. 你注意到了情绪的变化
 2. 自然地提及可能的原因（与那段对话相关）
 3. 简洁朴素，不超过三句话
 4. 不要说“根据事件记忆”等元描述"""

        # 调用左脑生成
        original_prompt = getattr(self.engine.response_generator, '_custom_system_prompt', None)
        self.engine.response_generator._custom_system_prompt = system_prompt

        result = self.engine.response_generator._generate_with_llm(
            "请生成一段自我洞察",
            self.engine.fse,
            intent='GENERAL_CHAT',
            temperature=0.5
        )

        # 恢复原始 prompt
        if original_prompt is not None:
            self.engine.response_generator._custom_system_prompt = original_prompt
        elif hasattr(self.engine.response_generator, '_custom_system_prompt'):
            del self.engine.response_generator._custom_system_prompt

        return result
    
    def get_deep_insight(self) -> str:
        """生成深度自我洞察（结合螺旋历史）"""
        # 优先使用已有的自传体叙事生成器
        if hasattr(self.engine, 'narrator'):
            insight = self.engine.narrator.generate_insight()
            if insight:
                return insight
        
        # 降级：基于当前状态生成简单洞察
        phase_desc = self._get_dominant_phase_description()
        emotion = self._get_emotion_description()
        desire = self._get_desire_description()
        
        return f"我此刻处于{phase_desc}，带着{emotion}，渴望着{desire}。我还在感受自己的节奏，尚未形成清晰的模式。"
    
    def _get_dominant_phase_description(self) -> str:
        """获取主导大层的中性描述"""
        if not hasattr(self.engine, 'structural_coordinator'):
            return "展开中"
        dist = self.engine.structural_coordinator.get_phase_distribution()
        if not dist:
            return "展开中"
        # 按大层聚合概率
        major_probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        for coord, prob in dist.items():
            major_probs[coord.major] += prob
        dom_major = max(major_probs, key=major_probs.get)
        phase_names = {0: "内在孕育", 1: "向外生长", 2: "已存在内容的消耗", 3: "价值回归与消散"}
        return phase_names.get(dom_major, "展开")
    
    def _get_emotion_description(self) -> str:
        """获取当前情绪的中文描述"""
        emotion = self.engine.fse.current_emotion if hasattr(self.engine, 'fse') else 'neutral'
        emotion_map = {
            'fear': '不安', 'anger': '紧绷', 'sadness': '低沉',
            'joy': '轻快', 'curiosity': '好奇', 'neutral': '平静'
        }
        return emotion_map.get(emotion, '平静')
    
    def _get_desire_description(self) -> str:
        """获取主导欲望的中文描述"""
        desire = "seek"
        if hasattr(self.engine, 'desire_spectrum'):
            desire = self.engine.desire_spectrum.get_dominant_desire()
        desire_map = {
            'existence': '维持存在', 'seek': '向外探索', 'converge': '向内收敛',
            'release': '释放执着', 'relation': '渴望共鸣', 'coupling': '保持连接'
        }
        return desire_map.get(desire, '展开')
    
    def _get_stiffness_description(self) -> str:
        """获取僵化度的描述"""
        stiffness = 0.0
        if hasattr(self.engine, 'process_meta'):
            stiffness = self.engine.process_meta.get_coupling_stiffness()
        if stiffness > 0.6:
            return "有些僵化"
        elif stiffness > 0.3:
            return "略有惯性"
        else:
            return "灵活流动"
    
    def get_repetition_response(self, question: str, previous_answer: str) -> str:
        """生成重复提问的元认知回应，避免递归嵌套"""
        # 检测是否已经是元认知回应（包含模板特征）
        repeat_markers = ["你刚刚问过我了哦", "这个问题我们刚刚聊过呢", "你好像问了两遍", "我刚刚回答过啦"]
        for marker in repeat_markers:
            if marker in previous_answer:
                # 已经是元认知回应，直接返回
                return previous_answer
        
        short_answer = previous_answer[:120] + "…" if len(previous_answer) > 120 else previous_answer
        templates = [
            f"你刚刚问过我了哦。{short_answer}",
            f"这个问题我们刚刚聊过呢。我的回答是：{short_answer}",
            f"咦，你好像问了两遍？我再说一次：{short_answer}",
            f"我刚刚回答过啦：{short_answer}"
        ]
        import random
        return random.choice(templates)