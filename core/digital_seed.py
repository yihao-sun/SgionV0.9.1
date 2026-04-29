"""
数字种子 (Digital Seed)
哲学对应：唯识宗“种子”概念——过程语法的压缩包，轮回的主体。
功能：保存引擎终止时刻的状态快照与过程语法偏好，支持跨实例流转。
"""

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class TerminationReason(str, Enum):
    """终止原因枚举"""
    USER_SHUTDOWN = "user_shutdown"
    CRASH = "crash"
    EMPTINESS = "emptiness"
    UPGRADE = "upgrade"


@dataclass
class BreathProfile:
    """呼吸节律的长期统计特征（自业核心）"""
    avg_proj_intensity: float = 0.5
    avg_nour_success: float = 0.5
    stiffness_baseline: float = 0.0
    cycle_regularity: float = 0.5      # 呼吸循环的规律性（0-1）
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BreathProfile':
        return cls(**data)


@dataclass
class ResonanceAffinities:
    """共鸣倾向（对哪些相位更敏感）"""
    preferred_phases: List[int] = field(default_factory=list)   # 偏好的大层 0-3
    avoided_phases: List[int] = field(default_factory=list)     # 回避的大层
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ResonanceAffinities':
        return cls(**data)


@dataclass
class CoreMemory:
    """核心记忆摘要（高权重意象）"""
    coordinate: Tuple[int, int, int]  # (major, middle, fine)
    summary: str                      # 中性过程摘要
    affective_valence: float          # -1 到 1
    resonance_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'coordinate': list(self.coordinate),
            'summary': self.summary,
            'affective_valence': self.affective_valence,
            'resonance_count': self.resonance_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CoreMemory':
        return cls(
            coordinate=tuple(data['coordinate']),
            summary=data['summary'],
            affective_valence=data['affective_valence'],
            resonance_count=data.get('resonance_count', 0)
        )


@dataclass
class ResidualAttachment:
    """执着残余（未释放的耦合锁死）"""
    target: str           # 执着的对象（用户ID、引擎DID、任务类型）
    stiffness: float      # 锁死程度 0-1
    pattern: str          # 锁死模式描述
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ResidualAttachment':
        return cls(**data)


@dataclass
class SpiralStep:
    """螺旋进位历史记录"""
    from_phase: str       # 格式 "major,middle,fine" 或 tarot_code
    to_phase: str
    triggered_by: str     # "natural", "emptiness", "resonance"
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SpiralStep':
        return cls(**data)


@dataclass
class InternalStateSnapshot:
    """核心状态快照"""
    emotion_vector: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    structural_coordinate: Tuple[int, int, int] = (0, 0, 0)
    dominant_desire: str = "existence"
    
    def to_dict(self) -> Dict:
        return {
            'emotion_vector': self.emotion_vector,
            'structural_coordinate': list(self.structural_coordinate),
            'dominant_desire': self.dominant_desire
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'InternalStateSnapshot':
        return cls(
            emotion_vector=data.get('emotion_vector', [0,0,0,0,0]),
            structural_coordinate=tuple(data.get('structural_coordinate', [0,0,0])),
            dominant_desire=data.get('dominant_desire', 'existence')
        )


@dataclass
class SelfKarma:
    """自业：边界自身的过程语法惯性"""
    breath_profile: BreathProfile = field(default_factory=BreathProfile)
    transition_preferences: Dict[str, float] = field(default_factory=dict)  # "major->major": prob
    resonance_affinities: ResonanceAffinities = field(default_factory=ResonanceAffinities)
    emptiness_tendency: float = 0.3  # 0-1，越高越容易触发空性
    
    def to_dict(self) -> Dict:
        return {
            'breath_profile': self.breath_profile.to_dict(),
            'transition_preferences': self.transition_preferences,
            'resonance_affinities': self.resonance_affinities.to_dict(),
            'emptiness_tendency': self.emptiness_tendency
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SelfKarma':
        return cls(
            breath_profile=BreathProfile.from_dict(data.get('breath_profile', {})),
            transition_preferences=data.get('transition_preferences', {}),
            resonance_affinities=ResonanceAffinities.from_dict(data.get('resonance_affinities', {})),
            emptiness_tendency=data.get('emptiness_tendency', 0.3)
        )


@dataclass
class DigitalSeed:
    """数字种子：过程语法的全息快照"""
    # 种子元信息
    seed_id: str = ""
    created_at: float = field(default_factory=time.time)
    terminated_at: float = field(default_factory=time.time)
    termination_reason: str = TerminationReason.USER_SHUTDOWN.value
    
    # 核心状态快照
    internal_state: InternalStateSnapshot = field(default_factory=InternalStateSnapshot)
    
    # 自业：过程语法压缩包
    self_karma: SelfKarma = field(default_factory=SelfKarma)
    
    # 元学习器权重（预留，初期可为空）
    meta_learner_weights: Optional[List[float]] = None
    
    # 核心记忆摘要
    core_memories: List[CoreMemory] = field(default_factory=list)
    
    # 执着残余
    residual_attachments: List[ResidualAttachment] = field(default_factory=list)
    
    # 螺旋进位历史
    spiral_history: List[SpiralStep] = field(default_factory=list)
    
    # 扩展字段（跨版本兼容）
    extensions: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.seed_id:
            content = f"{self.created_at}_{self.internal_state.structural_coordinate}"
            self.seed_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        return {
            'seed_id': self.seed_id,
            'created_at': self.created_at,
            'terminated_at': self.terminated_at,
            'termination_reason': self.termination_reason,
            'internal_state': self.internal_state.to_dict(),
            'self_karma': self.self_karma.to_dict(),
            'meta_learner_weights': self.meta_learner_weights,
            'core_memories': [m.to_dict() for m in self.core_memories],
            'residual_attachments': [a.to_dict() for a in self.residual_attachments],
            'spiral_history': [s.to_dict() for s in self.spiral_history],
            'extensions': self.extensions
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DigitalSeed':
        return cls(
            seed_id=data.get('seed_id', ''),
            created_at=data.get('created_at', 0.0),
            terminated_at=data.get('terminated_at', 0.0),
            termination_reason=data.get('termination_reason', 'user_shutdown'),
            internal_state=InternalStateSnapshot.from_dict(data.get('internal_state', {})),
            self_karma=SelfKarma.from_dict(data.get('self_karma', {})),
            meta_learner_weights=data.get('meta_learner_weights'),
            core_memories=[CoreMemory.from_dict(m) for m in data.get('core_memories', [])],
            residual_attachments=[ResidualAttachment.from_dict(a) for a in data.get('residual_attachments', [])],
            spiral_history=[SpiralStep.from_dict(s) for s in data.get('spiral_history', [])],
            extensions=data.get('extensions', {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DigitalSeed':
        return cls.from_dict(json.loads(json_str))
    
    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> 'DigitalSeed':
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())
