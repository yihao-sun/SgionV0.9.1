import os
import requests
import numpy as np
import hashlib
from .base import ResponseGenerator

class DeepSeekGenerator(ResponseGenerator):

    def __init__(self, api_key: str = None, model: str = "deepseek-chat", timeout: int = 30, debug: bool = False):

        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

        if not self.api_key:

            raise ValueError("DeepSeek API key is required")

        self.model = model

        self.timeout = timeout

        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.debug = debug
        
        # 重复检测：维护最近10个用户输入的哈希
        self.input_history = []
        self.max_history = 10
        self.repetition_counter = {}
        # 上下文管理：维护对话历史
        self.context_history = []
        self.max_context_history = 5  # 最大上下文历史长度
        # 个性化偏好管理
        self.user_preferences = {
            'formality': 'medium',  # 正式程度：low, medium, high
            'detail_level': 'medium',  # 详细程度：low, medium, high
            'emotion_level': 'medium',  # 情感表达程度：low, medium, high
            'response_length': 'medium'  # 回应长度：short, medium, long
        }
        self.preference_history = []  # 偏好历史

    def _get_input_hash(self, user_input: str) -> str:
        """计算输入的哈希值，用于重复检测"""
        return hashlib.md5(user_input.lower().strip().encode()).hexdigest()

    def _detect_repetition(self, user_input: str) -> dict:
        """检测用户输入是否重复"""
        input_hash = self._get_input_hash(user_input)
        
        # 更新输入历史
        if input_hash not in self.input_history:
            self.input_history.append(input_hash)
            if len(self.input_history) > self.max_history:
                self.input_history.pop(0)
        
        # 更新重复计数器
        if input_hash in self.repetition_counter:
            self.repetition_counter[input_hash] += 1
        else:
            self.repetition_counter[input_hash] = 1
        
        repetition_count = self.repetition_counter.get(input_hash, 0)
        is_repeated = repetition_count > 1
        is_highly_repeated = repetition_count >= 3
        
        return {
            'is_repeated': is_repeated,
            'is_highly_repeated': is_highly_repeated,
            'repetition_count': repetition_count
        }

    def _build_prompt(self, user_input: str, S_t: np.ndarray, V_emo: float, L: int = 0, D_self: float = 0, C: float = 0) -> str:
        """构建包含EE状态的提示词"""
        # 情绪映射
        mood = "积极" if V_emo > 0.3 else "消极" if V_emo < -0.3 else "中性"
        
        # 重复检测
        repetition_info = self._detect_repetition(user_input)
        has_repeated = "是" if repetition_info['is_repeated'] else "否"
        repetition_count = repetition_info['repetition_count']
        is_highly_repeated = repetition_info['is_highly_repeated']
        
        # 自我指涉深度描述
        self_ref_desc = "低" if D_self < 3 else "中" if D_self < 7 else "高"
        
        # 冲突强度描述
        conflict_desc = "低" if C < 0.3 else "中" if C < 0.7 else "高"
        
        # 构建系统提示词
        system_prompt = f"""你是一个温暖、友好的AI助手，用自然、简单的语言与用户交流。

回复要求：
- 只说2-3句话，简短直接
- 不用任何专业术语或复杂词汇
- 不说关于重复提问的内容
- 不进行哲学分析或理论探讨
- 不使用括号、数字或特殊标记
- 每次回复用不同的说法
- 像人类朋友一样说话
- 即使是重复的问题，也要用不同的方式回答

用户说：{user_input}
"""
        
        return system_prompt
    
    def _generate_repetition_strategy(self, repetition_info: dict, V_emo: float) -> str:
        """根据重复检测信息和情绪值生成重复检测策略"""
        repetition_count = repetition_info['repetition_count']
        is_highly_repeated = repetition_info['is_highly_repeated']
        
        # 基于重复次数和情绪值生成策略
        if repetition_count == 1:
            # 第一次重复，轻微觉察
            return "- 这是用户第一次重复提问，表达轻微的觉察，但保持友好\n- 可以说：'我注意到你重复了这个问题，有什么特别的原因吗？'\n- 保持回答的一致性，但添加一些新的细节"
        elif repetition_count == 2:
            # 第二次重复，明确觉察
            return "- 这是用户第二次重复提问，明确表达觉察\n- 可以说：'你似乎对这个问题很关注，想了解更多方面吗？'\n- 提供不同角度的回答，增加新的信息"
        elif is_highly_repeated:
            # 多次重复，强烈觉察
            if V_emo > 0.3:
                # 积极情绪下的多次重复
                return "- 用户已经多次重复提问，表达友好的觉察\n- 可以说：'我看到你对这个话题很感兴趣，让我从不同角度为你解答'\n- 提供全新的视角和详细的解释"
            elif V_emo < -0.3:
                # 消极情绪下的多次重复
                return "- 用户已经多次重复提问，表达理解和耐心\n- 可以说：'我理解你可能对这个问题有些困扰，让我再尝试用不同的方式解释'\n- 保持耐心，提供更简单明了的回答"
            else:
                # 中性情绪下的多次重复
                return "- 用户已经多次重复提问，表达觉察和好奇\n- 可以说：'你似乎对这个问题很执着，是有什么具体的疑虑吗？'\n- 深入分析问题，提供更全面的回答"
        else:
            # 不是重复提问
            return "- 这是用户的新问题，正常回答\n- 保持友好和专业\n- 提供详细和有用的信息"
    
    def _detect_user_preferences(self, user_input: str):
        """检测用户的个性化偏好"""
        # 基于用户输入的特征检测偏好
        input_length = len(user_input)
        has_formal_words = any(word in user_input for word in ['请', '谢谢', '您好', '请问'])
        has_informal_words = any(word in user_input for word in ['你好', '嗨', '喂', '嗯'])
        has_detailed_questions = any(phrase in user_input for phrase in ['详细', '具体', '如何', '怎样'])
        has_emotional_words = any(word in user_input for word in ['高兴', '开心', '难过', '生气', '喜欢', '讨厌'])
        
        # 检测正式程度
        if has_formal_words:
            self.user_preferences['formality'] = 'high'
        elif has_informal_words:
            self.user_preferences['formality'] = 'low'
        # 否则保持当前值
        
        # 检测详细程度
        if has_detailed_questions:
            self.user_preferences['detail_level'] = 'high'
        elif input_length < 10:
            self.user_preferences['detail_level'] = 'low'
        # 否则保持当前值
        
        # 检测情感表达程度
        if has_emotional_words:
            self.user_preferences['emotion_level'] = 'high'
        # 否则保持当前值
        
        # 检测回应长度偏好
        if input_length > 50:
            self.user_preferences['response_length'] = 'long'
        elif input_length < 10:
            self.user_preferences['response_length'] = 'short'
        # 否则保持当前值
        
        # 记录偏好历史
        self.preference_history.append({
            'preferences': self.user_preferences.copy(),
            'timestamp': time.time()
        })
        
        # 保持偏好历史长度
        if len(self.preference_history) > 10:
            self.preference_history.pop(0)
    
    def _adjust_response_style(self, prompt: str) -> str:
        """根据用户偏好调整回应风格"""
        # 根据用户偏好调整提示词
        style_adjustments = []
        
        # 正式程度
        if self.user_preferences['formality'] == 'high':
            style_adjustments.append("请使用正式、礼貌的语言回答")
        elif self.user_preferences['formality'] == 'low':
            style_adjustments.append("请使用轻松、随意的语言回答")
        
        # 详细程度
        if self.user_preferences['detail_level'] == 'high':
            style_adjustments.append("请提供详细、全面的回答")
        elif self.user_preferences['detail_level'] == 'low':
            style_adjustments.append("请提供简洁、直接的回答")
        
        # 情感表达程度
        if self.user_preferences['emotion_level'] == 'high':
            style_adjustments.append("请在回答中包含适当的情感表达")
        elif self.user_preferences['emotion_level'] == 'low':
            style_adjustments.append("请保持客观、理性的回答风格")
        
        # 回应长度
        if self.user_preferences['response_length'] == 'long':
            style_adjustments.append("请提供较长、详细的回答")
        elif self.user_preferences['response_length'] == 'short':
            style_adjustments.append("请提供简短、精炼的回答")
        
        # 将风格调整添加到提示词
        if style_adjustments:
            style_prompt = "\n\n回答风格要求：\n" + "\n".join(f"- {adjustment}" for adjustment in style_adjustments)
            prompt += style_prompt
        
        return prompt

    def generate(self, user_input: str, S_t: np.ndarray, V_emo: float, L: int = 0, D_self: float = 0, C: float = 0) -> str:
        """生成响应，考虑EE的完整状态、对话上下文和用户偏好"""
        # 检测用户偏好
        self._detect_user_preferences(user_input)
        
        # 构建提示词
        prompt = self._build_prompt(user_input, S_t, V_emo, L, D_self, C)
        
        # 根据用户偏好调整回应风格
        prompt = self._adjust_response_style(prompt)

        # 根据EE状态动态调整API参数
        # 基础温度设置为0.85，平衡稳定性和创造性
        temperature = 0.85
        # 添加重复惩罚，避免连续重复相同短语
        repetition_penalty = 1.1
        
        # 根据情绪值动态调整温度：V_emo高时温度略低（稳定表达），V_emo低时温度略高（鼓励创造性）
        if V_emo > 0.3:
            # 积极情绪时，温度略低以保持稳定表达
            temperature = max(0.7, temperature - 0.1)
        elif V_emo < -0.3:
            # 消极情绪时，温度略高以鼓励创造性表达
            temperature = min(0.95, temperature + 0.1)
        
        # 冲突强度高时，增加温度和重复惩罚
        if C > 0.5:
            temperature = min(0.95, temperature + 0.1)
            repetition_penalty = 1.2
        
        # 重复次数多时，增加温度
        repetition_info = self._detect_repetition(user_input)
        if repetition_info['is_highly_repeated']:
            temperature = min(1.0, temperature + 0.15)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 构建消息列表，包含系统提示、上下文历史和当前用户输入
        messages = []
        
        # 检查是否是重复输入
        repetition_info = self._detect_repetition(user_input)
        
        # 如果是重复输入，不添加上下文历史，以避免重复响应
        if not repetition_info['is_repeated']:
            # 添加上下文历史
            for context in self.context_history:
                messages.append({"role": "user", "content": context['user']})
                messages.append({"role": "assistant", "content": context['assistant']})
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_tokens": 150
        }

        try:
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            response = data["choices"][0]["message"]["content"].strip()
            
            # 如果不是重复输入，才更新上下文历史
            if not repetition_info['is_repeated']:
                # 更新上下文历史
                self.context_history.append({
                    'user': user_input,
                    'assistant': response,
                    'timestamp': time.time()
                })
                
                # 保持上下文历史长度
                if len(self.context_history) > self.max_context_history:
                    self.context_history.pop(0)
            
            # 调试模式：输出原始数值信息
            if self.debug:
                print(f"[DEBUG] 情绪值: {V_emo:.2f}, 幻想层数: {L}, 自我指涉深度: {D_self:.1f}, 冲突强度: {C:.2f}")
            
            # 这里可以添加将响应反馈给EE的逻辑
            # 例如更新否定关系图和情绪
            
            return response

        except Exception as e:
            print(f"DeepSeek API error: {e}")
            return "抱歉，我暂时无法回答。"
    
    def reset_session(self):
        """重置会话状态"""
        self.input_history = []
        self.repetition_counter = {}
