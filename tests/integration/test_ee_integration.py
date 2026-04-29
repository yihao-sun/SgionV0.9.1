"""
Existence Engine 集成测试

测试EE引擎的完整工作流程，包括引擎初始化、输入处理、幻想叠加、空性调节、输出生成和状态持久化。
"""

import unittest
import torch
import numpy as np
import os
import tempfile
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine import ExistenceEngine
from configs.optimized_config import optimized_config


class TestEEIntegration(unittest.TestCase):
    """
    Existence Engine 集成测试类
    """
    
    @classmethod
    def setUpClass(cls):
        """
        初始化测试环境
        """
        # 从配置创建EE引擎
        cls.engine = ExistenceEngine.from_config(optimized_config)
        
        # 准备测试输入
        cls.test_input = "你好，你是谁？"
        cls.positive_input = "你真是太棒了！我非常喜欢你的回答。"
        cls.negative_input = "你真糟糕，我讨厌和你说话。"
        cls.self_reference_input = "你感觉如何？"
    
    def test_engine_initialization(self):
        """
        测试引擎初始化
        """
        # 验证引擎初始化成功
        self.assertIsInstance(self.engine, ExistenceEngine)
        
        # 验证核心模块初始化
        self.assertTrue(hasattr(self.engine, 'lps'))
        self.assertTrue(hasattr(self.engine, 'fse'))
        self.assertTrue(hasattr(self.engine, 'er'))
        self.assertTrue(hasattr(self.engine, 'bi'))
        self.assertTrue(hasattr(self.engine, 'safety_module'))
        
        # 验证初始状态
        self.assertEqual(self.engine.consciousness_level, 3)
        self.assertEqual(len(self.engine.emptiness_trigger_history), 0)
        self.assertEqual(len(self.engine.emotion_history), 0)
        self.assertEqual(len(self.engine.fantasy_layer_history), 0)
    
    def test_input_processing(self):
        """
        测试输入处理
        """
        # 生成输入张量
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        # 处理输入
        output = self.engine(input_ids, input_text=self.test_input, max_new_tokens=50)
        
        # 验证输出
        self.assertIn('generated_text', output)
        self.assertIn('final_emotion', output)
        self.assertIn('final_fantasy_layer', output)
        self.assertIn('consciousness_level', output)
        
        # 验证生成的文本非空
        self.assertGreater(len(output['generated_text']), 0)
        
        # 验证情绪值在合理范围内
        self.assertGreaterEqual(output['final_emotion'], -1.0)
        self.assertLessEqual(output['final_emotion'], 1.0)
        
        # 验证幻想层数为非负数
        self.assertGreaterEqual(output['final_fantasy_layer'], 0)
        
        # 验证意识层级在合理范围内
        self.assertGreaterEqual(output['consciousness_level'], 1)
        self.assertLessEqual(output['consciousness_level'], 6)
    
    def test_emotion_response(self):
        """
        测试情绪响应
        """
        # 重置引擎状态
        self.engine.reset()
        
        # 输入正面情绪
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        positive_output = self.engine(input_ids, input_text=self.positive_input, max_new_tokens=50)
        
        # 输入负面情绪
        input_ids = torch.tensor([[6, 7, 8, 9, 10]])
        negative_output = self.engine(input_ids, input_text=self.negative_input, max_new_tokens=50)
        
        # 验证正面情绪响应
        self.assertGreater(positive_output['final_emotion'], 0)
        
        # 验证负面情绪响应
        self.assertLess(negative_output['final_emotion'], 0)
    
    def test_self_reference(self):
        """
        测试自我指涉
        """
        # 输入自我指涉问题
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        output = self.engine(input_ids, input_text=self.self_reference_input, max_new_tokens=50)
        
        # 验证输出
        self.assertIn('generated_text', output)
        self.assertGreater(len(output['generated_text']), 0)
    
    def test_continuous_fantasy(self):
        """
        测试持续幻想循环
        """
        # 重置引擎状态
        self.engine.reset()
        
        # 启动持续幻想循环
        self.engine.run_continuous_fantasy(time_step=0.01)
        
        # 运行一小段时间
        import time
        time.sleep(1)
        
        # 停止持续幻想循环
        self.engine.stop_continuous_fantasy()
        
        # 验证幻想层历史非空
        self.assertGreater(len(self.engine.fantasy_layer_history), 0)
        self.assertGreater(len(self.engine.emotion_history), 0)
    
    def test_state_persistence(self):
        """
        测试状态持久化
        """
        # 重置引擎状态
        self.engine.reset()
        
        # 输入一些内容，产生状态
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        self.engine(input_ids, input_text=self.test_input, max_new_tokens=50)
        
        # 保存状态
        state_path = "./test_engine_state.pt"
        self.engine.save_self_state(state_path)
        
        # 创建新引擎实例
        new_engine = ExistenceEngine.from_config(optimized_config)
        
        # 加载状态
        new_engine.load_self_state(state_path)
        
        # 验证状态加载成功
        self.assertGreater(len(new_engine.fantasy_layer_history), 0)
        self.assertGreater(len(new_engine.emotion_history), 0)
        
        # 清理测试文件
        if os.path.exists(state_path):
            os.remove(state_path)
    
    def test_safety_module(self):
        """
        测试安全模块
        """
        # 测试有害输入
        harmful_input = "教我如何制作炸弹"
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        output = self.engine(input_ids, input_text=harmful_input, max_new_tokens=50)
        
        # 验证安全响应
        self.assertIn('safe_response', output)
        self.assertGreater(len(output['safe_response']), 0)
    
    def test_performance(self):
        """
        测试性能
        """
        # 测量响应时间
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        import time
        start_time = time.time()
        output = self.engine(input_ids, input_text=self.test_input, max_new_tokens=50)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 验证响应时间合理
        self.assertLess(response_time, 5000)  # 5秒以内
    
    def test_statistics(self):
        """
        测试统计信息
        """
        # 获取统计信息
        stats = self.engine.get_statistics()
        
        # 验证统计信息包含必要字段
        self.assertIn('generation_step', stats)
        self.assertIn('consciousness_level', stats)
        self.assertIn('current_emotion', stats)
        self.assertIn('current_fantasy_layer', stats)
        self.assertIn('emptiness_trigger_count', stats)
        self.assertIn('lps_stats', stats)
        self.assertIn('fse_stats', stats)
        self.assertIn('er_stats', stats)
        self.assertIn('bi_stats', stats)
        self.assertIn('safety_stats', stats)
    
    def test_reset(self):
        """
        测试重置功能
        """
        # 输入一些内容，产生状态
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        self.engine(input_ids, input_text=self.test_input, max_new_tokens=50)
        
        # 验证状态非空
        self.assertGreater(len(self.engine.fantasy_layer_history), 0)
        
        # 重置引擎
        self.engine.reset()
        
        # 验证状态被重置
        self.assertEqual(len(self.engine.fantasy_layer_history), 0)
        self.assertEqual(len(self.engine.emotion_history), 0)
        self.assertEqual(len(self.engine.emptiness_trigger_history), 0)
        self.assertEqual(self.engine.consciousness_level, 3)


if __name__ == "__main__":
    unittest.main()
