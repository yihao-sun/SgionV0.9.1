"""
LPS模块与FSE的集成测试

测试LPS模块与FSE（幻想叠加引擎）的集成功能，包括在场/不在场标记和反哺机制。
"""

import unittest
import numpy as np
from lps import PotentialSpace, Potential


class TestLPSIntegration(unittest.TestCase):
    """
    LPS模块与FSE集成的测试类
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
    
    def test_present_absent_marking(self):
        """
        测试在场/不在场标记的正确性
        """
        # 模拟FSE调用LPS获取可能性场
        k = 5
        possibilities = self.lps.query(self.query_vector, k=k)
        
        # 模拟FSE选择在场可能性（通常选择势能最高的）
        present = possibilities[0]
        absent = possibilities[1:]
        
        # 验证在场可能性是势能最高的
        max_potency = max(p.potency for p in possibilities)
        self.assertEqual(present.potency, max_potency)
        
        # 验证不在场标记集合的大小
        self.assertEqual(len(absent), k-1)
        
        # 验证所有不在场可能性的势能都小于等于在场可能性
        for p in absent:
            self.assertLessEqual(p.potency, present.potency)
    
    def test_low_potency_sampling_for_fse(self):
        """
        测试FSE从低势能区域采样的功能
        """
        # 模拟FSE在需要探索时调用低势能采样
        threshold = 0.2
        
        # 采样多个低势能可能性
        low_potency_samples = []
        for _ in range(10):
            sample = self.lps.sample_low_potency(self.query_vector, threshold=threshold)
            low_potency_samples.append(sample)
        
        # 验证所有采样结果的相似度低于阈值
        for sample in low_potency_samples:
            similarity = np.dot(self.query_vector, sample.mu) / (
                np.linalg.norm(self.query_vector) * np.linalg.norm(sample.mu)
            )
            self.assertLess(similarity, threshold)
        
        # 验证采样结果具有多样性
        unique_samples = set(tuple(s.mu.tolist()) for s in low_potency_samples)
        self.assertGreater(len(unique_samples), 1)
    
    def test_potency_update_after_realization(self):
        """
        测试可能性被实现后的势能更新
        """
        # 获取一个可能性
        possibilities = self.lps.query(self.query_vector, k=1)
        original_potential = possibilities[0]
        original_potency = original_potential.potency
        
        # 模拟FSE实现该可能性（反哺机制）
        # 这里我们通过添加一个具有相同mu但更高potency的可能性来模拟
        updated_potential = Potential(
            mu=original_potential.mu,
            sigma=original_potential.sigma,
            potency=original_potency * 1.5  # 实现后势能增加
        )
        
        # 更新LPS中的可能性
        self.lps.update([updated_potential])
        
        # 重新查询，验证是否能找到高势能的可能性
        updated_possibilities = self.lps.query(self.query_vector, k=5)
        
        # 验证至少有一个可能性的势能大于原始势能
        has_higher_potency = any(p.potency > original_potency for p in updated_possibilities)
        self.assertTrue(has_higher_potency)
    
    def test_noise_effect_on_exploration(self):
        """
        测试噪声对探索的影响
        """
        # 两次查询，一次不添加噪声，一次添加噪声
        results_no_noise = self.lps.query(self.query_vector, k=5, add_noise=False)
        results_with_noise = self.lps.query(self.query_vector, k=5, add_noise=True)
        
        # 获取两次查询的可能性ID（通过mu的哈希值）
        ids_no_noise = [hash(tuple(r.mu.tolist())) for r in results_no_noise]
        ids_with_noise = [hash(tuple(r.mu.tolist())) for r in results_with_noise]
        
        # 验证添加噪声后返回的可能性有所不同
        # 由于噪声的存在，排序可能会发生变化
        self.assertNotEqual(ids_no_noise, ids_with_noise)


if __name__ == "__main__":
    unittest.main()
