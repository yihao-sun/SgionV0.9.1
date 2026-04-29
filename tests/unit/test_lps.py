"""
LPS模块验证测试

根据功能验证方案，测试LPS模块的核心功能、边界条件、性能指标和与FSE的集成。
"""

import unittest
import numpy as np
import time
import os
import tempfile
from lps import PotentialSpace, Potential


class TestLPS(unittest.TestCase):
    """
    LPS模块的单元测试类
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
    
    def test_basic_query(self):
        """
        测试基本查询功能
        """
        # 测试查询功能
        k = 5
        results = self.lps.query(self.query_vector, k=k, min_potency=0.1, add_noise=False)
        
        # 验证返回数量
        self.assertEqual(len(results), k)
        
        # 验证按势能降序排列
        potencies = [r.potency for r in results]
        for i in range(k-1):
            self.assertGreaterEqual(potencies[i], potencies[i+1])
        
        # 验证每个结果包含必要字段
        for result in results:
            self.assertTrue(hasattr(result, 'mu'))
            self.assertTrue(hasattr(result, 'sigma'))
            self.assertTrue(hasattr(result, 'potency'))
            self.assertEqual(result.mu.shape, (self.embedding_dim,))
            self.assertEqual(result.sigma.shape, (self.embedding_dim,))
    
    def test_noise_addition(self):
        """
        测试噪声添加功能
        """
        # 两次查询，一次不添加噪声，一次添加噪声
        results_no_noise = self.lps.query(self.query_vector, k=5, add_noise=False)
        results_with_noise = self.lps.query(self.query_vector, k=5, add_noise=True)
        
        # 验证两次结果的势能不同
        potencies_no_noise = [r.potency for r in results_no_noise]
        potencies_with_noise = [r.potency for r in results_with_noise]
        
        # 至少有一个结果的势能不同
        self.assertTrue(any(abs(p1 - p2) > 1e-6 for p1, p2 in zip(potencies_no_noise, potencies_with_noise)))
    
    def test_low_potency_sampling(self):
        """
        测试低势能采样功能
        """
        threshold = 0.2
        samples = []
        
        # 采样100次
        for _ in range(100):
            sample = self.lps.sample_low_potency(self.query_vector, threshold=threshold)
            samples.append(sample)
        
        # 验证所有采样结果的相似度低于阈值
        similarities = []
        for sample in samples:
            similarity = np.dot(self.query_vector, sample.mu) / (
                np.linalg.norm(self.query_vector) * np.linalg.norm(sample.mu)
            )
            similarities.append(similarity)
            self.assertLess(similarity, threshold)
        
        # 验证采样的多样性
        unique_samples = set(tuple(sample.mu.tolist()) for sample in samples)
        self.assertGreater(len(unique_samples), 1)
    
    def test_persistence(self):
        """
        测试持久化功能
        """
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # 保存LPS状态
            self.lps.save(temp_path)
            
            # 创建新实例并加载状态
            lps2 = PotentialSpace(
                embedding_dim=self.embedding_dim,
                max_possibilities=self.max_possibilities
            )
            lps2.load(temp_path)
            
            # 测试加载后的查询结果
            results1 = self.lps.query(self.query_vector, k=5, add_noise=False)
            results2 = lps2.query(self.query_vector, k=5, add_noise=False)
            
            # 验证结果相似
            for r1, r2 in zip(results1, results2):
                np.testing.assert_allclose(r1.mu, r2.mu)
                self.assertAlmostEqual(r1.potency, r2.potency, places=6)
        
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if os.path.exists(temp_path + '.faiss'):
                os.unlink(temp_path + '.faiss')
    
    def test_query_performance(self):
        """
        测试查询性能
        """
        # 测量1000次查询的平均耗时
        start_time = time.perf_counter()
        for _ in range(1000):
            self.lps.query(self.query_vector, k=10)
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / 1000
        print(f"平均查询耗时: {avg_time * 1000:.2f} ms")
        
        # 验证查询延迟 < 1ms (CPU)
        self.assertLess(avg_time, 0.001)  # 1ms = 0.001 seconds
    
    def test_dynamic_expansion(self):
        """
        测试动态扩展功能
        """
        # 获取当前可能性数量
        initial_count = self.lps.get_statistics()['num_possibilities']
        
        # 添加新的可能性
        new_possibilities = []
        for i in range(100):
            mu = np.random.randn(self.embedding_dim)
            mu = mu / np.linalg.norm(mu)
            sigma = np.ones(self.embedding_dim) * 0.1
            potency = np.random.uniform(0.1, 1.0)
            new_possibilities.append(Potential(mu, sigma, potency))
        
        self.lps.update(new_possibilities)
        
        # 验证可能性数量增加
        final_count = self.lps.get_statistics()['num_possibilities']
        self.assertGreater(final_count, initial_count)
    
    def test_potency_pruning(self):
        """
        测试势能修剪功能
        """
        # 获取当前可能性数量
        initial_count = self.lps.get_statistics()['num_possibilities']
        
        # 触发修剪
        self.lps._prune_low_potency(keep_ratio=0.5)
        
        # 验证可能性数量减少
        final_count = self.lps.get_statistics()['num_possibilities']
        self.assertLess(final_count, initial_count)
        self.assertLessEqual(final_count, int(initial_count * 0.5))
    
    def test_statistics(self):
        """
        测试统计信息功能
        """
        stats = self.lps.get_statistics()
        
        # 验证统计信息包含必要字段
        self.assertIn('num_possibilities', stats)
        self.assertIn('avg_potency', stats)
        self.assertIn('potency_std', stats)
        self.assertIn('embedding_dim', stats)
        self.assertIn('max_possibilities', stats)
        
        # 验证统计信息合理
        self.assertGreater(stats['num_possibilities'], 0)
        self.assertGreaterEqual(stats['avg_potency'], 0)
        self.assertGreaterEqual(stats['potency_std'], 0)
        self.assertEqual(stats['embedding_dim'], self.embedding_dim)
        self.assertEqual(stats['max_possibilities'], self.max_possibilities)


if __name__ == "__main__":
    unittest.main()
