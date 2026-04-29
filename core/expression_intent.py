from dataclasses import dataclass, field
from typing import List

@dataclass
class ExpressionIntent:
    facts: str = ""                         # 事实骨架
    imagery_fragments: List[str] = field(default_factory=list)
    emotion: str = "neutral"
    major_phase: int = 1
    freshness: float = 0.5
    persona: str = "息觀"
    forbidden_phrases: List[str] = field(default_factory=lambda: [
        "作为AI", "我没有情感", "无法理解", "据我所知", "一般认为",
        "尼采", "康德", "海德格尔", "萨特"  # 禁止提及非核心层哲学家
    ])