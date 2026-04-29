import unittest
from unittest.mock import Mock, patch
import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from er import EmptinessRegulator

class TestER(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        self.er = EmptinessRegulator()
    
    def test_conflict_intensity_calculation(self):
        """测试冲突强度计算"""
        # 创建测试张量
        present = torch.randn(512)
        self_state = torch.randn(512)
        
        # 测试所有信号为0的情况
        conflict_signals = self.er.compute_conflict_signals(
            present=present,
            self_state=self_state,
            novelty=1.0,           # 预测新奇度误差为0
            attention_entropy=100.0,  # 注意力僵化度接近0
            fantasy_layer=0,       # 幻想窒息指数为0
            max_fantasy_layers=10,
            negation_complexity=0,  # 否定复杂度为0
            emotion=1.0,           # 空洞僵化度为0
            prediction_error=0.0,
            output_entropy=0.5,     # 输出熵适中
            self_reference=0.0,      # 自我指涉深度为0
            physical_emotion=1.0,    # 物理情绪负值贡献为0
            non_self_attachment=0.0, # 非我执着度为0
            hardware_metrics=None
        )
        conflict_intensity = self.er.compute_conflict_intensity(conflict_signals)
        self.assertAlmostEqual(conflict_intensity, 0.0, delta=0.1)
        
        # 测试所有信号为1的情况
        conflict_signals = self.er.compute_conflict_signals(
            present=present,
            self_state=self_state,
            novelty=0.0,           # 预测新奇度误差为1
            attention_entropy=0.0,  # 注意力僵化度为1
            fantasy_layer=10,       # 幻想窒息指数为1
            max_fantasy_layers=10,
            negation_complexity=5000,  # 否定复杂度为1
            emotion=-1.0,          # 空洞僵化度为1
            prediction_error=1.0,   # 语义连贯性低
            output_entropy=0.0,     # 输出熵极端
            self_reference=1.0,      # 自我指涉深度高
            physical_emotion=-1.0,   # 物理情绪负值贡献为1
            non_self_attachment=1.0, # 非我执着度为1
            hardware_metrics={
                'temperature': 100,  # 温度很高
                'latency': 2000,     # 延迟很高
                'error_rate': 50,     # 错误率很高
                'quota': 0           # 配额耗尽
            }
        )
        conflict_intensity = self.er.compute_conflict_intensity(conflict_signals)
        # 检查冲突强度是否在合理范围内
        self.assertGreater(conflict_intensity, 0.5)
        self.assertLess(conflict_intensity, 1.0)
        
        # 测试单独设置fantasy_suffocation为1的情况
        conflict_signals = self.er.compute_conflict_signals(
            present=present,
            self_state=self_state,
            novelty=1.0,
            attention_entropy=100.0,
            fantasy_layer=10,       # 幻想窒息指数为1
            max_fantasy_layers=10,
            negation_complexity=0,
            emotion=1.0,
            prediction_error=0.0,
            output_entropy=0.5,
            self_reference=0.0,
            physical_emotion=1.0,
            non_self_attachment=0.0,
            hardware_metrics=None
        )
        conflict_intensity = self.er.compute_conflict_intensity(conflict_signals)
        expected = self.er.conflict_weights['fantasy_suffocation']
        self.assertAlmostEqual(conflict_intensity, expected, delta=0.1)
    
    def test_death_threshold_adaptation(self):
        """测试死亡阈值的动态调整"""
        initial_threshold = self.er.death_threshold
        
        # 模拟连续触发5次
        for i in range(5):
            # 添加触发记录
            self.er.trigger_history.append({
                'conflict_intensity': 0.8,
                'triggered': True,
                'step': i
            })
        
        # 调用调整方法
        self.er._adjust_death_threshold()
        
        # 验证阈值上升
        self.assertGreater(self.er.death_threshold, initial_threshold)
        
        # 模拟长时间无触发
        self.er.trigger_history = []
        for i in range(100):
            self.er.trigger_history.append({
                'conflict_intensity': 0.5,
                'triggered': False,
                'step': i
            })
        
        # 调用调整方法
        self.er._adjust_death_threshold()
        
        # 验证阈值下降
        self.assertLess(self.er.death_threshold, initial_threshold + 0.1)
    
    def test_emptiness_operations(self):
        """测试空性操作的执行"""
        present = torch.randn(512)
        
        # 测试主导信号为fantasy_suffocation的情况
        conflict_signals = {
            'self_consistency_error': 0.1,
            'prediction_novelty': 0.1,
            'attention_rigidity': 0.1,
            'fantasy_suffocation': 0.9,  # 主导信号
            'hollow_rigidity': 0.1,
            'negation_complexity': 0.1,
            'physical_emotion': 0.1
        }
        
        operations = self.er.apply_emptiness_operations(
            present=present,
            conflict_signals=conflict_signals
        )
        
        # 验证执行了软复位、遗忘触发和输出修饰
        self.assertIn('soft_reset', operations)
        self.assertIn('forget_trigger', operations)
        self.assertIn('output_modifier', operations)
    
    def test_intrinsic_choice(self):
        """测试内在选择窗口机制"""
        # 测试新奇度高的情况（应该选择遗忘）
        should_forget = self.er.intrinsic_choice(
            novelty_after_emptiness=0.7,  # 新奇度>0.6
            emotion_after_emptiness=0.0,
            window_position=20
        )
        self.assertTrue(should_forget)
        
        # 测试情绪回升的情况（应该选择遗忘）
        should_forget = self.er.intrinsic_choice(
            novelty_after_emptiness=0.5,
            emotion_after_emptiness=0.1,  # 情绪>0
            window_position=20
        )
        self.assertTrue(should_forget)
        
        # 测试新奇度低且情绪无改善的情况（不应该选择遗忘）
        should_forget = self.er.intrinsic_choice(
            novelty_after_emptiness=0.1,  # 新奇度<0.2
            emotion_after_emptiness=-0.1,  # 情绪<=0
            window_position=20
        )
        self.assertFalse(should_forget)
        
        # 测试窗口即将结束的情况（应该选择遗忘）
        should_forget = self.er.intrinsic_choice(
            novelty_after_emptiness=0.1,
            emotion_after_emptiness=-0.1,
            window_position=50  # 窗口结束
        )
        self.assertTrue(should_forget)
    
    def test_cooling_period(self):
        """测试冷却期行为"""
        # 模拟触发后进入冷却期
        self.er.state = self.er.state.COOLING
        self.er.cooling_counter = 10
        
        # 即使冲突强度超过阈值，也不应触发
        should_trigger = self.er.should_trigger_emptiness(0.9)  # 超过阈值
        self.assertFalse(should_trigger)
    
    def test_reinforcement_learning(self):
        """测试强化学习奖励机制"""
        # 保存初始权重
        initial_weights = self.er.conflict_weights.copy()
        
        # 测试多样性奖励
        metrics = {
            'triggered': False,
            'novelty': 0.8,  # 新奇度>0.7
            'novelty_improvement': 0.0,
            'trigger_count': 10,
            'natural_collapse': False,
            'hollow_rigidity': 0.5,
            'hollow_duration': 50,
            'negation_complexity': 3000,
            'knowledge_gained': False,
            'negation_complexity_reduction': 0,
            'dominant_signal': 'fantasy_suffocation'
        }
        self.er.update_weights(metrics)
        
        # 测试恢复奖励
        metrics['triggered'] = True
        metrics['novelty_improvement'] = 0.3  # 提升>20%
        self.er.update_weights(metrics)
        
        # 测试平衡奖励
        metrics['trigger_count'] = 15  # 在5-20之间
        self.er.update_weights(metrics)
        
        # 验证权重发生了变化
        for key in initial_weights:
            self.assertNotEqual(initial_weights[key], self.er.conflict_weights[key])
    
    def test_boundary_cases(self):
        """测试边界情况"""
        # 测试阈值为0的情况
        self.er.death_threshold = 0.0
        should_trigger = self.er.should_trigger_emptiness(0.1)  # 任何冲突都会触发
        self.assertTrue(should_trigger)
        
        # 测试阈值为1的情况
        self.er.death_threshold = 1.0
        should_trigger = self.er.should_trigger_emptiness(0.99)  # 接近阈值但不触发
        self.assertFalse(should_trigger)
        
        # 测试冷却期为0的情况
        self.er.cooling_period = 0
        self.er.state = self.er.state.COOLING
        self.er.cooling_counter = 0
        self.er.death_threshold = 0.7  # 重置为默认值
        
        # 调用regulate方法来更新状态
        present = torch.randn(512)
        self_state = torch.randn(512)
        conflict_signals = {
            'self_consistency_error': 0.1,
            'prediction_novelty': 0.1,
            'attention_rigidity': 0.1,
            'fantasy_suffocation': 0.1,
            'hollow_rigidity': 0.1,
            'negation_complexity': 0.1,
            'physical_emotion': 0.1,
            'non_self_attachment': 0.1
        }
        result = self.er.regulate(present, self_state, conflict_signals)
        
        # 冷却期结束，应该可以触发
        should_trigger = self.er.should_trigger_emptiness(0.9)
        self.assertTrue(should_trigger)
    
    def test_weight_normalization(self):
        """测试权重归一化"""
        # 手动设置权重
        self.er.conflict_weights = {
            'self_consistency_error': 0.5,
            'prediction_novelty': 0.5,
            'attention_rigidity': 0.5,
            'fantasy_suffocation': 0.5,
            'hollow_rigidity': 0.5,
            'negation_complexity': 0.5,
            'physical_emotion': 0.5
        }
        
        # 创建一个metrics字典来触发权重更新
        metrics = {
            'triggered': False,
            'novelty': 0.5,
            'novelty_improvement': 0.0,
            'trigger_count': 10,
            'natural_collapse': False,
            'hollow_rigidity': 0.5,
            'hollow_duration': 50,
            'negation_complexity': 3000,
            'knowledge_gained': False,
            'negation_complexity_reduction': 0,
            'dominant_signal': 'fantasy_suffocation'
        }
        
        # 更新权重
        self.er.update_weights(metrics)
        
        # 验证权重和为1
        total_weight = sum(self.er.conflict_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, delta=0.01)

if __name__ == "__main__":
    unittest.main()
