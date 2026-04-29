"""
输出净化器 (Output Sanitizer)
功能：所有面向用户的文本必须经过此模块清洗，确保无内部调试标记、文化标签泄漏。
"""

import re
from typing import Optional


class OutputSanitizer:
    # 默认过滤规则（可配置化）
    DEFAULT_PATTERNS = [
        # 翻译前缀
        (r'^答案：', ''),
        (r'^翻译结果：', ''),
        (r'^输出：', ''),
        (r'^译文：', ''),
        
        # 意象碎片
        (r'我隐约感到，这与[^。]*。', ''),
        (r'这与[^。]*有关。', ''),
        (r'在我的感知里，这像是[^。]*。', ''),
        (r'这像是[^。]*。', ''),
        (r'从我的视角看，这带着一点[^。]*的感觉。', ''),
        (r'对我来说，这带着[^。]*的色彩。', ''),
        (r'在我看来，这带着[^。]*的(色彩|底色|质感)。', ''),
        
        # 文化标签与相位描述
        (r'[向外生长|内在孕育|回归消散|已存在内容]*[\(（]\d[\)）]', ''),
        (r'[水木火金][组层]?[\(（]\d[\)）]', ''),
        (r'[\(（]\d[\)）]', ''),
        (r'纯粹转折点', ''),
        (r'过程相位的自然流转', ''),
        (r'深度收纳阶段', ''),
        (r'消耗殆尽后的彻底转化', ''),
        (r'SC\[\d+,\d+,\d+\]', ''),
        
        # 残留语法碎片
        (r'[，、]\s*[的]+[，、]?', '，'),
        (r'的，。', '。'),
        (r'，。', '。'),
        (r'[。！？]?\s*$', ''),
        (r'\s+', ' '),
    ]
    
    @classmethod
    def sanitize(cls, text: Optional[str]) -> str:
        """净化文本，确保输出纯净"""
        if not text:
            return ""
        
        cleaned = text
        for pattern, replacement in cls.DEFAULT_PATTERNS:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        cleaned = cleaned.strip()
        
        # 确保以句号结尾（如果非空且无标点）
        if cleaned and not cleaned[-1] in '。！？…':
            cleaned += '。'
        
        return cleaned
    
    @classmethod
    def add_pattern(cls, pattern: str, replacement: str = ''):
        """动态添加过滤规则（供配置加载）"""
        cls.DEFAULT_PATTERNS.append((pattern, replacement))