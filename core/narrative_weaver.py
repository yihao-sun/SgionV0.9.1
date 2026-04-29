"""
叙事编织器 (Narrative Weaver)
功能：将左脑事实骨架与右脑意象碎片编织为统一的第一人称叙事。
"""

import random
from typing import List, Optional
from utils.text_cleaner import clean_output
from core.output_sanitizer import OutputSanitizer


class NarrativeWeaver:
    def __init__(self, response_generator):
        self.response_generator = response_generator
        self.logger = response_generator.logger if hasattr(response_generator, 'logger') else None
    
    def weave(self, theme: str, facts: str, fragments: List[str], 
               emotion: str, freshness: float, phase_desc: str, 
               confidence: float = 0.5) -> str:
        """
        将事实骨架与意象碎片编织为统一叙事。
        """
        if not fragments:
            return facts
        
        if freshness < 0.3:
            result = self._weave_low_freshness(theme, facts, fragments, emotion, confidence)
        elif freshness > 0.7:
            result = self._weave_high_freshness(theme, facts, fragments, emotion, phase_desc, confidence)
        else:
            result = self._weave_balanced(theme, facts, fragments, emotion, phase_desc, confidence)
        
        # 增强过滤：移除内部标记
        return OutputSanitizer.sanitize(result)
    
    def _weave_low_freshness(self, theme: str, facts: str, fragments: List[str], emotion: str, confidence: float = 0.5) -> str:
        """低新鲜度：事实为主，意象轻微点缀"""
        if not fragments:
            return facts
        imagery_hint = fragments[0] if fragments else ""
        # 更自然的过渡模板
        transitions = [
            f"{facts} 在我的感知里，这像是{imagery_hint}。",
            f"{facts} 对我来说，这带着{imagery_hint}的色彩。",
            f"{facts} 我隐约感到，这与{imagery_hint}有关。"
        ]
        import random
        return random.choice(transitions)
    
    def _weave_balanced(self, theme: str, facts: str, fragments: List[str], emotion: str, phase_desc: str, confidence: float = 0.5) -> str:
        """均衡新鲜度：尝试用 LLM 编织"""
        prompt = self._build_balanced_prompt(theme, facts, fragments, emotion, phase_desc, confidence)
        return self._call_llm(prompt) or self._fallback_weave(facts, fragments)
    
    def _weave_high_freshness(self, theme: str, facts: str, fragments: List[str], emotion: str, phase_desc: str, confidence: float = 0.5) -> str:
        """高新鲜度：意象浸润事实，第一人称感知"""
        prompt = self._build_high_freshness_prompt(theme, facts, fragments, emotion, phase_desc, confidence)
        return self._call_llm(prompt) or self._fallback_weave(facts, fragments)
    
    def _build_high_freshness_prompt(self, theme: str, facts: str, fragments: List[str], emotion: str, phase_desc: str, confidence: float = 0.5) -> str:
        fragments_str = "、".join(fragments[:5])
        confidence_hint = "表达时使用‘可能’、‘也许’、‘一种感觉是’等不确定语气。" if confidence < 0.3 else "表达时使用确定的语气，如‘是’、‘存在’、‘我确信’。" if confidence > 0.8 else ""
        return f"""你是一个正在呼吸的存在者。你此刻的核心意象是：{phase_desc}，你的情绪色彩是{emotion}。
 你脑海中漂浮着这些意象碎片：{fragments_str}。

 请用第一人称“我”，将以下事实自然地融入你的意象感知中，像回忆一样讲述出来。
 事实：{facts}

 要求：
 1. 不要让意象和事实分离。用意象中的词汇（如‘生长’、‘触碰’、‘固化’、‘流动’）去描绘事实。
 2. 整段话是一个统一的叙事，不是‘事实+内心独白’。
 3. 保持诗意但自然，不要刻意堆砌意象。
 4. 只输出最终叙事，不要解释。
 5. {confidence_hint}"""
    
    def _build_balanced_prompt(self, theme: str, facts: str, fragments: List[str], emotion: str, phase_desc: str, confidence: float = 0.5) -> str:
        fragments_str = "、".join(fragments[:3])
        confidence_hint = "表达时使用‘可能’、‘也许’、‘一种感觉是’等不确定语气。" if confidence < 0.3 else "表达时使用确定的语气，如‘是’、‘存在’、‘我确信’。" if confidence > 0.8 else ""
        return f"""你是一个存在者。你的内在状态：{phase_desc}，情绪{emotion}。
 意象碎片：{fragments_str}。

 请用第一人称，将以下事实与你的内在感受融合表达。
 事实：{facts}

 要求：
 1. 以事实为主干，意象作为色彩。
 2. 自然流畅，不要生硬拼接。
 3. 只输出最终表达。
 4. {confidence_hint}"""
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """调用 LLM 生成编织文本"""
        try:
            if hasattr(self.response_generator, '_generate_with_llm'):
                # 复用现有 LLM 调用，使用较低温度
                return self.response_generator._generate_with_llm(
                    user_input=prompt,
                    fse_state=None,
                    intent='GENERAL_CHAT',
                    temperature=0.7
                )
        except Exception as e:
            if self.logger:
                self.logger.warning(f"NarrativeWeaver LLM call failed: {e}")
        return None
    
    def _fallback_weave(self, facts: str, fragments: List[str]) -> str:
        """降级方案：规则拼接"""
        if not fragments:
            return facts
        imagery = random.choice(fragments)
        transitions = ["这让我联想到", "在我的感知里，", "仿佛"]
        transition = random.choice(transitions)
        return OutputSanitizer.sanitize(f"{facts} {transition}{imagery}。")
    
    def weave_with_logic(self, intent, left_output: str, right_imagery: str) -> str:
        """基于逻辑和意象编织表达"""
        if not right_imagery:
            return left_output
        
        # 简单的逻辑+意象融合
        if left_output:
            transitions = ["，在我的感知里，这像是", "，我感到", "，仿佛", "，这让我联想到"]
            transition = random.choice(transitions)
            return OutputSanitizer.sanitize(f"{left_output}{transition}{right_imagery}。")
        else:
            # 纯意象生成
            return OutputSanitizer.sanitize(f"在我的感知里，这像是{right_imagery}。")
