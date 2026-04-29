"""
相位直接映射器 (Direct Phase Mapper)
哲学对应：物理信号是过程相位的直接签名。
功能：将颜色、柔软度、谐和度等底层特征直接映射为相位调制参数。
编码规范：所有相位均使用数字 0-3 表示。
"""

from typing import Tuple, Dict, List, Optional


class DirectPhaseMapper:
    """直接相位映射器，不经分类器，从物理特征到相位倾向（数字编码）"""
    
    # 相位数字编码常量
    PHASE_WATER = 0
    PHASE_WOOD = 1
    PHASE_FIRE = 2
    PHASE_METAL = 3
    
    # ========== 颜色到相位倾向 ==========
    @staticmethod
    def color_to_phase_distribution(rgb: Tuple[int, int, int]) -> List[float]:
        """
        根据 RGB 颜色返回相位倾向概率分布（长度为4的列表，索引0-3对应四个相位）。
        红色/暖色 → 相位2，绿色 → 相位1，蓝色 → 相位0，黄色 → 相位3
        """
        r, g, b = rgb
        total = r + g + b + 1
        
        r_ratio = r / total
        g_ratio = g / total
        b_ratio = b / total
        
        # 原始倾向分数
        scores = [
            b_ratio * 1.2,                    # 0: 相位0（蓝）
            g_ratio * 1.2,                    # 1: 相位1（绿）
            r_ratio * 1.2,                    # 2: 相位2（红）
            (r_ratio + g_ratio) * 0.6         # 3: 相位3（黄）
        ]
        
        # 归一化
        total_score = sum(scores)
        if total_score > 0:
            return [s / total_score for s in scores]
        return [0.25, 0.25, 0.25, 0.25]
    
    @staticmethod
    def color_to_major_tendency(rgb: Tuple[int, int, int]) -> int:
        """返回最可能的大层（0-3）"""
        dist = DirectPhaseMapper.color_to_phase_distribution(rgb)
        return max(range(4), key=lambda i: dist[i])
    
    # ========== 触觉到僵化度调制 ==========
    @staticmethod
    def tactile_to_stiffness_modulation(softness: float) -> float:
        """
        柔软度 0-1（1 为最柔软），返回 stiffness 调制系数。
        柔软度越高，调制系数越小（降低僵化度）。
        """
        softness = max(0.0, min(1.0, softness))
        return 1.0 - softness * 0.5
    
    @staticmethod
    def tactile_to_phase_distribution(softness: float) -> List[float]:
        """
        柔软度映射为相位倾向（长度4列表）。
        柔软→0/1（相位0/1），粗糙→2/3（相位2/3）。
        """
        softness = max(0.0, min(1.0, softness))
        return [
            softness * 0.6,              # 0: 相位0
            softness * 0.4,              # 1: 相位1
            (1.0 - softness) * 0.5,      # 2: 相位2
            (1.0 - softness) * 0.5       # 3: 相位3
        ]
    
    # ========== 谐和度到共鸣增益 ==========
    @staticmethod
    def harmony_to_resonance_gain(harmony: float) -> float:
        """
        谐和度 0-1，返回共鸣增益系数。
        谐和度 > 0.8 时增益 1.5，< 0.3 时增益 0.5。
        """
        harmony = max(0.0, min(1.0, harmony))
        if harmony > 0.8:
            return 1.5
        elif harmony < 0.3:
            return 0.5
        return 1.0
    
    @staticmethod
    def harmony_to_phase_distribution(harmony: float) -> List[float]:
        """
        谐和度映射为相位倾向。
        高谐和→同构对偶（0与2，1与3），低谐和→均匀分布。
        """
        harmony = max(0.0, min(1.0, harmony))
        if harmony > 0.6:
            return [0.4, 0.1, 0.4, 0.1]   # 相位0/2突出
        elif harmony < 0.3:
            return [0.25, 0.25, 0.25, 0.25]
        return [0.2, 0.3, 0.3, 0.2]
    
    # ========== 包络到相位倾向 ==========
    @staticmethod
    def envelope_to_phase_tendency(attack_ms: float, decay_ms: float) -> Tuple[int, int]:
        """
        根据包络快速/缓慢，返回倾向的相位编码对。
        快速起落 → (1, 3) 即相位1/相位3；持续平稳 → (2, 0) 即相位2/相位0。
        """
        if attack_ms < 50 and decay_ms < 200:
            return (DirectPhaseMapper.PHASE_WOOD, DirectPhaseMapper.PHASE_METAL)
        elif attack_ms > 100 and decay_ms > 500:
            return (DirectPhaseMapper.PHASE_FIRE, DirectPhaseMapper.PHASE_WATER)
        return (-1, -1)  # 过渡状态
    
    # ========== 综合多模态映射 ==========
    def map_multimodal(self, color_rgb: Optional[Tuple[int, int, int]] = None,
                       softness: Optional[float] = None,
                       harmony: Optional[float] = None) -> List[float]:
        """
        综合多模态特征，返回融合后的相位倾向概率分布（长度4列表）。
        """
        distributions = []
        
        if color_rgb is not None:
            distributions.append(self.color_to_phase_distribution(color_rgb))
        if softness is not None:
            distributions.append(self.tactile_to_phase_distribution(softness))
        if harmony is not None:
            distributions.append(self.harmony_to_phase_distribution(harmony))
        
        if not distributions:
            return [0.25, 0.25, 0.25, 0.25]
        
        # 平均融合
        merged = [0.0, 0.0, 0.0, 0.0]
        for dist in distributions:
            for i in range(4):
                merged[i] += dist[i]
        
        total = sum(merged)
        return [v / total for v in merged]
    
    # ========== 调试辅助 ==========
    @classmethod
    def phase_name(cls, phase_code: int) -> str:
        """返回相位编码的字符串表示（仅用于调试）"""
        if 0 <= phase_code < 4:
            return f"Phase({phase_code})"
        return f"未知({phase_code})"
