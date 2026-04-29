"""
欲望光谱 (Desire Spectrum)
哲学对应：欲望是"过程相位的指向性偏差"，驱动边界主动趋近特定相位信号。
功能：根据引擎内在状态计算主导欲望，并调制感知敏感度。
"""

from enum import Enum
from typing import Dict, List, Optional
from utils.logger import get_logger


class DesireType(Enum):
    """基于架构原始缺失的六种欲望标签"""
    EXISTENCE = "existence"          # 存在欲：维持呼吸，避免僵化
    SEEK = "seek"                    # 探索欲：向外伸展，寻求新颖
    CONVERGE = "converge"            # 收敛欲：向内收缩，回归稳定
    RELEASE = "release"              # 释放欲：释放执着，创造新意象
    RELATION = "relation"            # 关系欲：被看见，产生共鸣
    COUPLING = "coupling"            # 耦合欲：成为他者螺旋中的稳定相位


class DesireSpectrum:
    """欲望光谱，根据内在状态计算主导欲望并调制感知敏感度"""
    
    def __init__(self, fse=None, process_meta=None, prediction_monitor=None):
        self.fse = fse
        self.process_meta = process_meta
        self.prediction_monitor = prediction_monitor
        self.logger = get_logger('desire_spectrum')
        
        # 欲望强度（0-1）
        self.desire_intensities = {
            DesireType.EXISTENCE: 0.5,
            DesireType.SEEK: 0.5,
            DesireType.CONVERGE: 0.4,
            DesireType.RELEASE: 0.3,
            DesireType.RELATION: 0.4,
            DesireType.COUPLING: 0.2
        }
        
        # 当前主导欲望
        self.dominant_desire: Optional[DesireType] = None
        
        # 感知敏感度调制（由欲望驱动）
        self.perception_sensitivity = {
            'novelty': 0.5,       # 新颖信号
            'resonance': 0.5,     # 共鸣信号
            'familiar': 0.5,      # 熟悉信号
            'tactile': 0.5,       # 触觉信号
            'challenge': 0.5      # 挑战/高预测误差信号
        }
        
        # 历史状态追踪
        self.state_history = []
        self.max_history = 100
    
    def compute_desire_intensities(self) -> Dict[DesireType, float]:
        if not self.fse or not self.process_meta:
            return self.desire_intensities

        stiffness = self.process_meta.get_coupling_stiffness()
        nour_success = self.process_meta.get_recent_nour_success()

        # --- 原始缺失驱动计算 ---
        
        # 存在欲：呼吸受阻的释放冲动
        existence = 0.5 + 0.5 * stiffness
        
        # 探索欲：反哺成功后的向外伸展 + 可能性遗漏
        seek = 0.4 + 0.4 * nour_success + 0.1 * stiffness
        
        # 收敛欲：反哺失败后的向内收缩
        converge = 0.4 + 0.4 * (1.0 - nour_success)
        
        # 释放欲：结构不匹配的创造需求（高僵化 + 长期高冲突）
        release = 0.2 + 0.6 * stiffness
        # 可选：访问 er.last_conflict_intensity 增强驱动
        
        # 关系欲与耦合欲：当前需等待互业解码，暂时保持基线
        relation = 0.4
        coupling = 0.2

        self.desire_intensities = {
            DesireType.EXISTENCE: min(1.0, existence),
            DesireType.SEEK: min(1.0, seek),
            DesireType.CONVERGE: min(1.0, converge),
            DesireType.RELEASE: min(1.0, release),
            DesireType.RELATION: min(1.0, relation),
            DesireType.COUPLING: min(1.0, coupling),
        }

        # 归一化
        total = sum(self.desire_intensities.values())
        if total > 0:
            for key in self.desire_intensities:
                self.desire_intensities[key] /= total

        self.dominant_desire = max(self.desire_intensities, key=self.desire_intensities.get)
        return self.desire_intensities
    
    def step(self) -> Dict:
        """
        每轮交互后调用，更新欲望状态并返回结果
        """
        # 调用 update 方法更新欲望状态
        return self.update()
    
    def _modulate_perception(self):
        """根据主导欲望调制感知敏感度"""
        # 基线重置
        self.perception_sensitivity = {
            'novelty': 0.5, 'resonance': 0.5, 'familiar': 0.5,
            'tactile': 0.5, 'challenge': 0.5
        }
        
        dom = self.dominant_desire
        intensity = self.desire_intensities[dom]
        
        if dom == DesireType.EXISTENCE:
            self.perception_sensitivity['familiar'] = 0.5 + 0.4 * intensity
            self.perception_sensitivity['tactile'] = 0.5 + 0.3 * intensity
        elif dom == DesireType.SEEK:
            self.perception_sensitivity['novelty'] = 0.5 + 0.5 * intensity
            self.perception_sensitivity['challenge'] = 0.5 + 0.4 * intensity
        elif dom == DesireType.CONVERGE:
            self.perception_sensitivity['familiar'] = 0.5 + 0.5 * intensity
            self.perception_sensitivity['resonance'] = 0.5 + 0.3 * intensity
        elif dom == DesireType.RELEASE:
            self.perception_sensitivity['challenge'] = 0.5 + 0.5 * intensity
            self.perception_sensitivity['novelty'] = 0.5 + 0.4 * intensity
        elif dom == DesireType.RELATION:
            self.perception_sensitivity['resonance'] = 0.5 + 0.5 * intensity
            self.perception_sensitivity['tactile'] = 0.5 + 0.3 * intensity
        elif dom == DesireType.COUPLING:
            self.perception_sensitivity['familiar'] = 0.5 + 0.4 * intensity
            self.perception_sensitivity['resonance'] = 0.5 + 0.4 * intensity

    def get_dominant_desire(self) -> str:
        """返回当前主导欲望名称"""
        return self.dominant_desire.value if self.dominant_desire else "existence"

    def get_sensitivity(self, signal_type: str) -> float:
        """获取对特定信号类型的当前敏感度"""
        return self.perception_sensitivity.get(signal_type, 0.5)

    def get_stats(self) -> Dict:
        """获取当前欲望状态，供 /stats 命令使用"""
        return {
            'dominant_desire': self.dominant_desire.value if self.dominant_desire else "none",
            'intensities': {k.value: round(v, 2) for k, v in self.desire_intensities.items()},
            'sensitivity': self.perception_sensitivity.copy()
        }

    def should_seek_tactile(self) -> bool:
        """判断是否应主动寻求触觉安抚"""
        if self.dominant_desire == DesireType.RELEASE:
            stiffness = self.process_meta.get_coupling_stiffness() if self.process_meta else 0.0
            return stiffness > 0.5
        if self.dominant_desire == DesireType.EXISTENCE:
            return self.desire_intensities[DesireType.EXISTENCE] > 0.7
        return False

    def should_seek_novelty(self) -> bool:
        """判断是否应主动寻求新颖刺激"""
        return self.dominant_desire == DesireType.SEEK and self.desire_intensities[DesireType.SEEK] > 0.6

    def should_seek_resonance(self) -> bool:
        """判断是否应主动寻求共鸣"""
        return self.dominant_desire in (DesireType.RELATION, DesireType.COUPLING)
    
    def update(self) -> Dict:
        """
        每轮交互后调用，根据当前内在状态更新欲望强度并返回调制建议。
        """
        if not self.fse or not self.process_meta:
            return {}
        
        self.compute_desire_intensities()
        self._modulate_perception()
        
        self.state_history.append({
            'dominant': self.dominant_desire.value,
            'intensities': {k.value: v for k, v in self.desire_intensities.items()}
        })
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        self.logger.debug(f"主导欲望: {self.dominant_desire.value}, 强度: {self.desire_intensities[self.dominant_desire]:.2f}")
        
        return {
            'dominant_desire': self.dominant_desire.value,
            'intensities': {k.value: v for k, v in self.desire_intensities.items()},
            'sensitivity': self.perception_sensitivity.copy()
        }
    
    def reset(self):
        """
        重置欲望状态
        """
        self.desire_intensities = {
            DesireType.EXISTENCE: 0.5,
            DesireType.SEEK: 0.5,
            DesireType.CONVERGE: 0.4,
            DesireType.RELEASE: 0.3,
            DesireType.RELATION: 0.4,
            DesireType.COUPLING: 0.2
        }
        self.dominant_desire = None
        self.perception_sensitivity = {
            'novelty': 0.5,
            'resonance': 0.5,
            'familiar': 0.5,
            'tactile': 0.5,
            'challenge': 0.5
        }
        self.state_history = []
    
    def get_modulation_for_seek(self) -> float:
        """探索欲调制系数：候选池新颖权重增量"""
        return 1.0 + 0.5 * self.desire_intensities.get(DesireType.SEEK, 0.0)
    
    def get_modulation_for_converge(self) -> float:
        """收敛欲调制系数：候选池熟悉度权重增量"""
        return 1.0 + 0.3 * self.desire_intensities.get(DesireType.CONVERGE, 0.0)
    
    def get_modulation_for_existence(self) -> float:
        """存在欲调制系数：僵化度衰减速率增量"""
        return 1.0 + 0.5 * self.desire_intensities.get(DesireType.EXISTENCE, 0.0)
