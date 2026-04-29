"""
颜色编码器 (Color Coder)
哲学对应：意象层概念文档第8节，将结构坐标和呼吸印记映射为颜色。
功能：根据 StructuralCoordinate 和 breathSignature 生成 HSV/RGB 颜色码。
"""

import colorsys
from typing import Tuple, Dict
from core.structural_coordinator import StructuralCoordinate

class ColorCoder:
    def __init__(self, config=None):
        self.config = config or {}
        # 基础色相映射（大层 0-3），校准为近等距分布
        self.base_hues = {
            0: 240.0,   # 水：蓝 (240°) —— 内在孕育，向内收敛
            1: 120.0,   # 土：绿 (120°) —— 向外成形，生长感
            2: 0.0,     # 火：红 (0°)   —— 已存在内容，消耗与温度
            3: 60.0,    # 风：黄 (60°)  —— 内容回流，消散与通透
        }
        # 中层微调幅度（度）- 保留但不再用于色相偏移，改为调整饱和度
        self.middle_hue_shift = self.config.get('color', {}).get('middle_hue_shift', 0.0)
        # 细微层微调幅度（度）- 保留但不再用于色相偏移，改为调整明度
        self.fine_hue_shift = self.config.get('color', {}).get('fine_hue_shift', 0.0)

    def compute_hsv(self, coord: StructuralCoordinate, breath: Dict[str, float]) -> Tuple[float, float, float]:
        """
        计算 HSV 值。
        大层决定色相，中层决定饱和度，细微层决定明度。
        呼吸印记进行微调。
        """
        # 1. 大层决定色相
        base_h = self.base_hues.get(coord.major, 0.0)
        hue = (base_h) % 360.0
        hue_norm = hue / 360.0

        # 2. 中层决定饱和度
        middle_saturation = {
            0: 0.20,   # 水组：极低饱和度——边界开放，被动接收
            1: 0.90,   # 土组：高饱和度——边界明确，结构清晰
            2: 0.65,   # 火组：中高饱和度——共鸣激活，有选择性
            3: 0.50,   # 风组：中低饱和度——价值排序，偏松散
        }
        saturation = middle_saturation.get(coord.middle, 0.5)

        # 3. 细微层决定明度
        fine_value = {
            0: 0.40,   # 水细微：低明度——被动接纳，能量较低
            1: 0.60,   # 土细微：中等明度——温和吸收，后续转化
            2: 0.85,   # 火细微：高明度——即时确认，主动消耗
            3: 0.70,   # 风细微：中高明度——快速回流，能量偏高
        }
        value = fine_value.get(coord.fine, 0.6)

        # 4. 呼吸印记微调
        stiffness = breath.get('stiffness', 0.0)
        # 僵化度高时饱和度略微降低，颜色更灰
        saturation = max(0.1, saturation - stiffness * 0.3)

        proj = breath.get('proj_intensity', 0.5)
        nour = breath.get('nour_success', 0.5)
        # 反哺高于投射时颜色更亮，投射高于反哺时颜色更暗
        value = max(0.2, min(1.0, value + 0.15 * (proj - nour)))

        return (hue_norm, max(0.0, min(1.0, saturation)), max(0.0, min(1.0, value)))

    def compute_rgb(self, coord: StructuralCoordinate, breath: Dict[str, float]) -> Tuple[int, int, int]:
        """计算 RGB 值（0-255 整数）"""
        h, s, v = self.compute_hsv(coord, breath)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))

    def compute_hex(self, coord: StructuralCoordinate, breath: Dict[str, float]) -> str:
        """返回十六进制颜色码，如 #4A7FDB"""
        r, g, b = self.compute_rgb(coord, breath)
        return f"#{r:02X}{g:02X}{b:02X}"
