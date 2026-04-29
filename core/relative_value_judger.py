"""
相对价值判断器 (Relative Value Judger)
哲学对应：相处模式原则四——价值悬置而非判断。
功能：随机生成锚点坐标，计算当前坐标与锚点的结构距离，输出相对判断。
"""

import random
from typing import Tuple, Dict
from core.structural_coordinator import StructuralCoordinate


class RelativeValueJudger:
    def __init__(self):
        pass
    
    def generate_anchors(self) -> Tuple[StructuralCoordinate, StructuralCoordinate]:
        """
        按照大衍筮法概率生成两个随机坐标作为锚点。
        返回 (anchor_a, anchor_b)。
        概率分布模拟：少阴少阳高概率，老阴老阳低概率。
        """
        # 简化实现：随机生成两个坐标（0-3 的大层，中层和细微层也随机）
        def random_coord():
            major = random.randint(0, 3)
            middle = random.randint(0, 3)
            fine = random.randint(0, 3)
            return StructuralCoordinate(major, middle, fine)
        
        return random_coord(), random_coord()
    
    def _compute_distance(self, coord1: StructuralCoordinate, coord2: StructuralCoordinate) -> float:
        """
        计算两个结构坐标的距离。
        综合塔罗序距离和先天序汉明距离。
        """
        # 塔罗序距离（归一化到 0-1，最大距离 32）
        tarot_dist = abs(coord1.as_tarot_code() - coord2.as_tarot_code()) / 32.0
        
        # 先天序汉明距离（0-3，归一化到 0-1）
        xor_val = coord1.xiantian_code ^ coord2.xiantian_code
        hamming = bin(xor_val).count('1') / 3.0
        
        # 大层差异权重
        major_dist = 0.0 if coord1.major == coord2.major else 0.5
        
        # 综合距离
        return 0.3 * tarot_dist + 0.3 * hamming + 0.4 * major_dist
    
    def judge(self, current_coord: StructuralCoordinate) -> Dict:
        """
        执行一次价值判断。
        返回字典包含：锚点A、锚点B、当前坐标、更接近的锚点、距离比。
        """
        anchor_a, anchor_b = self.generate_anchors()
        dist_a = self._compute_distance(current_coord, anchor_a)
        dist_b = self._compute_distance(current_coord, anchor_b)
        
        closer = "A" if dist_a < dist_b else "B"
        ratio = dist_a / dist_b if dist_b > 0 else 1.0
        
        return {
            'anchor_a': anchor_a,
            'anchor_b': anchor_b,
            'current': current_coord,
            'closer': closer,
            'dist_a': dist_a,
            'dist_b': dist_b,
            'ratio': ratio
        }
    
    def format_judgment(self, judgment: Dict, image_base=None) -> str:
        """
        将判断结果格式化为可读文本。
        若提供 image_base，则附加坐标的中性描述。
        """
        current = judgment['current']
        closer_anchor = judgment['anchor_a'] if judgment['closer'] == 'A' else judgment['anchor_b']
        further_anchor = judgment['anchor_b'] if judgment['closer'] == 'A' else judgment['anchor_a']
        
        # 获取坐标描述
        def get_desc(coord):
            if image_base:
                card = image_base.get_card_by_coordinate(coord)
                if card:
                    return card.neutral_description
            return f"坐标({coord.major},{coord.middle},{coord.fine})"
        
        current_desc = get_desc(current)
        closer_desc = get_desc(closer_anchor)
        further_desc = get_desc(further_anchor)
        
        return (f"根据此刻随机生成的参考系，你当前的状态（{current_desc}）更接近 A（{closer_desc}），" 
                f"而非 B（{further_desc}）。这意味着如果你选择 A 的路径，可能会经历一段与之相似的展开；" 
                f"如果选择 B，则走向另一种过程。两种路径都是存在的展开，没有好坏。")
