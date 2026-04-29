"""
概率漫步器 (Probabilistic Walker)
基于大衍筮法概率模型：少阳 5/16、少阴 7/16、老阳 3/16、老阴 1/16
"""
import random
from typing import Tuple
from core.structural_coordinate import StructuralCoordinate


class ProbabilisticWalker:
    def __init__(self, engine=None):
        self.engine = engine
        self.base_probs = {
            'smooth': 12/16,      # 少阳 + 少阴
            'opposite': 3/16,     # 老阳
            'leap': 1/16          # 老阴
        }
    
    def step(self, current_coord: StructuralCoordinate) -> StructuralCoordinate:
        """根据当前坐标和内在状态，执行一次概率漫步"""
        stiffness = 0.0
        if self.engine and hasattr(self.engine, 'process_meta'):
            stiffness = self.engine.process_meta.get_coupling_stiffness()
        
        # 动态调制：僵化度升高时增加跃迁概率
        probs = self._modulate_probs(stiffness)
        r = random.random()
        
        if r < probs['smooth']:
            return self._smooth_transition(current_coord)
        elif r < probs['smooth'] + probs['opposite']:
            return self._opposite_transition(current_coord)
        else:
            return self._leap_transition(current_coord)
    
    def _modulate_probs(self, stiffness: float) -> dict:
        probs = self.base_probs.copy()
        if stiffness > 0.5:
            # 僵化时提高跃迁概率
            delta = min(0.15, (stiffness - 0.5) * 0.3)
            probs['opposite'] += delta * 0.6
            probs['leap'] += delta * 0.4
            probs['smooth'] = 1.0 - probs['opposite'] - probs['leap']
        return probs
    
    def _smooth_transition(self, coord: StructuralCoordinate) -> StructuralCoordinate:
        """平滑转移：塔罗序 ±1"""
        new_tarot = (coord.as_tarot_code() + random.choice([-1, 1])) % 64
        return self._coord_by_tarot(new_tarot)
    
    def _opposite_transition(self, coord: StructuralCoordinate) -> StructuralCoordinate:
        """对偶转移：先天序对偶"""
        opposite_xiantian = 7 - coord.xiantian_code
        return self._coord_by_xiantian(opposite_xiantian)
    
    def _leap_transition(self, coord: StructuralCoordinate) -> StructuralCoordinate:
        """极性跃迁：大层循环跳跃"""
        new_major = (coord.major + 2) % 4
        return StructuralCoordinate(new_major, coord.middle, coord.fine)
    
    def _coord_by_tarot(self, tarot_code: int) -> StructuralCoordinate:
        major = tarot_code // 16
        middle = (tarot_code % 16) // 4
        fine = tarot_code % 4
        return StructuralCoordinate(major, middle, fine)
    
    def _coord_by_xiantian(self, xiantian_code: int) -> StructuralCoordinate:
        major = (xiantian_code >> 2) & 3
        middle = (xiantian_code >> 1) & 1
        fine = xiantian_code & 1
        return StructuralCoordinate(major, middle, fine)