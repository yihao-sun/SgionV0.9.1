"""
LPS模块反哺机制测试

测试LPS模块的反哺机制和否定关系关联功能。
"""

import unittest
import numpy as np
from lps import PotentialSpace, Potential


class TestLPSFeedback(unittest.TestCase):
    """
    LPS模块反哺机制的测试类
    """
    
    @classmethod
    def setUpClass(cls):
        """
        初始化测试环境
        """
        # 创建PotentialSpace实例
        cls.embedding_dim = 64
        cls.max_possibilities = 1000
        cls.lps = PotentialSpace(
            embedding_dim=cls.embedding_dim,
            max_possibilities=cls.max_possibilities
        )
        
        # 创建测试查询向量
        cls.query_vector = np.random.randn(cls.embedding_dim)
        cls.query_vector = cls.query_vector / np.linalg.norm(cls.query_vector)
        
        # 添加测试可能性
        cls._add_test_possibilities()
    
    @classmethod
    def _add_test_possibilities(cls):
        """
        添加测试可能性
        """
        test_possibilities = []
        
        # 添加高势能可能性
        for i in range(50):
            mu = np.random.randn(cls.embedding_dim)
            mu = mu / np.linalg.norm(mu)
            sigma = np.ones(cls.embedding_dim) * 0.1
            potency = np.random.uniform(0.7, 1.0)
            test_possibilities.append(Potential(mu, sigma, potency))
        
        # 添加低势能可能性
        for i in range(50):
            mu = np.random.randn(cls.embedding_dim)
            mu = mu / np.linalg.norm(mu)
            sigma = np.ones(cls.embedding_dim) * 0.1
            potency = np.random.uniform(0.1, 0.3)
            test_possibilities.append(Potential(mu, sigma, potency))
        
        cls.lps.update(test_possibilities)
    
    def test_feedback_from_non_self(self):
        """
        测试反哺机制中的低势能采样
        
        当FSE需要从"非我"中汲取新内容时，调用sample_low_potency()获取一个可能性并转化为"在场"。
        """
        # 从低势能区域采样一个可能性
        threshold = 0.2
        low_potential = self.lps.sample_low_potency(self.query_vector, threshold=threshold)
        
        # 记录原始势能
        original_potency = low_potential.potency
        print(f"原始势能: {original_potency:.4f}")
        
        # 验证采样的可能性势能低于阈值
        similarity = np.dot(self.query_vector, low_potential.mu) / (
            np.linalg.norm(self.query_vector) * np.linalg.norm(low_potential.mu)
        )
        self.assertLess(similarity, threshold)
        
        # 模拟FSE将该可能性转化为"在场"（反哺机制）
        # 通过创建一个具有相同mu但更高potency的新可能性来模拟
        feedback_potential = Potential(
            mu=low_potential.mu,
            sigma=low_potential.sigma,
            potency=original_potency * 2.0  # 势能增加
        )
        
        # 更新LPS
        self.lps.update([feedback_potential])
        
        # 遍历所有可能性，找到对应的可能性
        updated_potential = None
        for potential in self.lps.possibilities:
            if np.allclose(potential.mu, low_potential.mu):
                updated_potential = potential
                break
        
        # 验证找到了更新后的可能性
        self.assertIsNotNone(updated_potential)
        
        # 验证势能增加
        new_potency = updated_potential.potency
        print(f"更新后势能: {new_potency:.4f}")
        self.assertGreater(new_potency, original_potency)
    
    def test_negation_relation_association(self):
        """
        测试否定关系与LPS的关联
        
        否定关系图中的一个节点应与LPS中的某个可能性mu关联，当该否定关系被反哺时，LPS中对应mu的势能增加。
        """
        # 模拟否定关系图中的一个节点（如"非我不是加法规则"）
        # 我们创建一个特定的mu来表示"加法规则"
        addition_rule_mu = np.random.randn(self.embedding_dim)
        addition_rule_mu = addition_rule_mu / np.linalg.norm(addition_rule_mu)
        
        # 添加一个低势能的"加法规则"可能性（表示"非我不是加法规则"）
        negation_potential = Potential(
            mu=addition_rule_mu,
            sigma=np.ones(self.embedding_dim) * 0.1,
            potency=0.2  # 低势能，表示被否定
        )
        self.lps.update([negation_potential])
        
        # 记录原始势能
        original_potency = negation_potential.potency
        print(f"否定关系原始势能: {original_potency:.4f}")
        
        # 模拟模型学习了加法规则（反哺）
        # 通过创建一个具有相同mu但更高potency的新可能性来模拟
        learned_potential = Potential(
            mu=addition_rule_mu,
            sigma=np.ones(self.embedding_dim) * 0.1,
            potency=original_potency * 3.0  # 势能显著增加
        )
        
        # 更新LPS
        self.lps.update([learned_potential])
        
        # 遍历所有可能性，找到对应的可能性
        updated_potential = None
        for potential in self.lps.possibilities:
            if np.allclose(potential.mu, addition_rule_mu):
                updated_potential = potential
                break
        
        # 验证找到了更新后的可能性
        self.assertIsNotNone(updated_potential)
        
        # 验证势能增加
        new_potency = updated_potential.potency
        print(f"否定关系更新后势能: {new_potency:.4f}")
        self.assertGreater(new_potency, original_potency)


if __name__ == "__main__":
    unittest.main()
