"""
结构坐标 (Structural Coordinate)
定义存在引擎的 3 层 64 相位结构坐标系统。
"""
from typing import Tuple


class StructuralCoordinate:
    """三层结构坐标数据类"""
    TAIJI_CODE = -1
    
    def __init__(self, major: int, middle: int, fine: int):
        self.major = major      # 0/1/2/3（四个基本过程相位）或 -1（太极）
        self.middle = middle    # 0-3 或 -1（纯态）或 4（综合态）
        self.fine = fine        # 0-3 或 -1（无/纯/综合）
        
        # 计算先天序编码（若含-1则设为-1）
        if major == -1 or middle == -1 or fine == -1:
            self.xiantian_code = -1
        else:
            major_bit = (major % 2) << 2
            middle_bit = (middle % 2) << 1
            fine_bit = (fine % 2)
            self.xiantian_code = major_bit | middle_bit | fine_bit
    
    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.major, self.middle, self.fine)
    
    def as_tarot_code(self) -> int:
        """将三层坐标压缩为单一塔罗序编码 (0-63)，公式: major*16 + middle*4 + fine"""
        if self.major == -1 or self.middle == -1 or self.fine == -1:
            return -1
        return self.major * 16 + self.middle * 4 + self.fine
    
    def get_opposite_xiantian(self) -> int:
        """返回先天序的对立值"""
        if self.xiantian_code == -1:
            return -1
        return 7 - self.xiantian_code
    
    @staticmethod
    def is_taiji(code) -> bool:
        """判断是否为太极值"""
        return code == StructuralCoordinate.TAIJI_CODE
    
    def __repr__(self):
        def fmt(v):
            return "x" if v == -1 else str(v)
        return f"SC[{fmt(self.major)},{fmt(self.middle)},{fmt(self.fine)}](T:{self.as_tarot_code()}, X:{self.xiantian_code})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, StructuralCoordinate):
            return False
        return (
            self.major == other.major and
            self.middle == other.middle and
            self.fine == other.fine
        )
    
    def __hash__(self) -> int:
        return hash((self.major, self.middle, self.fine))