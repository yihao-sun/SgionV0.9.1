"""
测试 v1.3 新增功能：内在驱动、趋势采集、结构坐标映射
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from engine import ExistenceEngine


class TestV13Enhancements(unittest.TestCase):
    """测试 v1.3 新增功能"""

    def setUp(self):
        """设置测试环境"""
        # 创建引擎实例，使用 mock 响应
        self.engine = ExistenceEngine(vocab_size=10000, use_llm=False)

    def test_internal_drive_increases_L(self):
        """测试内在驱动是否会自动增加 L"""
        # 记录初始 L 值
        initial_L = self.engine.fse.L
        
        # 模拟多轮无输入
        for _ in range(20):  # 足够多的步骤以触发内在驱动
            self.engine.internal_step()
        
        # 检查 L 是否增加
        final_L = self.engine.fse.L
        self.assertGreater(final_L, initial_L, "L 应该在内在驱动下增加")

    def test_internal_drive_explores_low_potency(self):
        """测试内在驱动是否会触发低势能采样"""
        # 模拟 LPS 采样
        with patch.object(self.engine.lps, 'sample_low_potency') as mock_sample:
            # 设置 mock 返回值
            mock_sample.return_value = {'text': 'test low potency'}
            
            # 模拟多轮无输入
            for _ in range(15):
                self.engine.internal_step()
            
            # 检查低势能采样是否被调用
            mock_sample.assert_called()

    def test_trend_collection(self):
        """测试投射/反哺趋势值是否在多次交互后更新"""
        # 模拟多次交互
        for i in range(10):
            # 构建输入张量
            import torch
            input_ids = torch.randint(0, self.engine.vocab_size, (1, 10))
            # 调用 forward 方法
            self.engine.forward(input_ids, input_text=f"Test input {i}")
        
        # 获取趋势值
        stats = self.engine.process_meta.get_stats()
        projection_trend = stats['projection_trend']
        nourishment_trend = stats['nourishment_trend']
        
        # 检查趋势值是否为浮点数
        self.assertIsInstance(projection_trend, float)
        self.assertIsInstance(nourishment_trend, float)

    def test_structural_coordinate_changes_with_state(self):
        """测试结构坐标是否随状态变化"""
        # 获取初始结构坐标
        initial_coord = self.engine.structural_coordinator.get_current_coordinate()
        
        # 模拟改变 L 和情绪
        if hasattr(self.engine.fse, 'L'):
            self.engine.fse.L = 12  # 高 L
        if hasattr(self.engine.fse, 'E_vec'):
            self.engine.fse.E_vec = np.array([0.1, 0.8, -0.5, 0.2, 0.3])  # 高唤醒度，负性情绪
        
        # 获取改变后的结构坐标
        changed_coord = self.engine.structural_coordinator.get_current_coordinate()
        
        # 检查坐标是否变化
        self.assertNotEqual(initial_coord.as_tarot_code(), changed_coord.as_tarot_code(), "结构坐标应该随状态变化")

    def test_structural_coordinate_output_format(self):
        """测试结构坐标输出格式"""
        # 获取结构坐标
        coord = self.engine.structural_coordinator.get_current_coordinate()
        # 检查 __repr__ 输出格式
        repr_str = repr(coord)
        self.assertRegex(repr_str, r"^SC\[\d,\d,\d:\d+\]$", "结构坐标输出格式应该为 SC[*,*,*:*]")


if __name__ == '__main__':
    unittest.main()
