import pytest
from core.structural_coordinator import StructuralCoordinate
from core.color_coder import ColorCoder

def test_color_mapping_consistency():
    coder = ColorCoder()
    coord = StructuralCoordinate(1, 2, 1)
    breath = {'proj_intensity': 0.7, 'nour_success': 0.3, 'stiffness': 0.2}
    
    hex1 = coder.compute_hex(coord, breath)
    hex2 = coder.compute_hex(coord, breath)
    assert hex1 == hex2  # 相同输入应输出相同颜色

def test_hue_wraparound():
    coder = ColorCoder()
    # 测试色相超出 360 度时正确回绕
    coord = StructuralCoordinate(3, 3, 3)  # 金(60) + 微调，可能接近 360
    breath = {'proj_intensity': 0.5, 'nour_success': 0.5, 'stiffness': 0.0}
    h, s, v = coder.compute_hsv(coord, breath)
    assert 0.0 <= h <= 1.0

def test_saturation_bounds():
    coder = ColorCoder()
    coord = StructuralCoordinate(0, 0, 0)
    # stiffness 为 1 时饱和度应为 0
    breath = {'proj_intensity': 0.5, 'nour_success': 0.5, 'stiffness': 1.0}
    h, s, v = coder.compute_hsv(coord, breath)
    assert s == 0.0

def test_value_bounds():
    coder = ColorCoder()
    coord = StructuralCoordinate(0, 0, 0)
    breath = {'proj_intensity': 0.0, 'nour_success': 1.0, 'stiffness': 0.0}
    h, s, v = coder.compute_hsv(coord, breath)
    assert v >= 0.2  # 明度下限
