"""
Unit Tests for Stage 1 Separability Experiment
阶段一可分性实验单元测试
"""

import unittest
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from similarity_analyzer import CosineSimilarityCalculator, SeparabilityAnalyzer, SimilarityResult
from data_generator import Sample, AttackSampleGenerator


class TestCosineSimilarityCalculator(unittest.TestCase):
    """测试余弦相似度计算器"""

    def test_identical_vectors(self):
        """测试相同向量相似度为1"""
        v = np.array([1.0, 2.0, 3.0])
        sim = CosineSimilarityCalculator.compute(v, v)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_orthogonal_vectors(self):
        """测试正交向量相似度为0"""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        sim = CosineSimilarityCalculator.compute(v1, v2)
        self.assertAlmostEqual(sim, 0.0, places=5)

    def test_opposite_vectors(self):
        """测试相反向量相似度为-1"""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = -v1
        sim = CosineSimilarityCalculator.compute(v1, v2)
        self.assertAlmostEqual(sim, -1.0, places=5)

    def test_batch_computation(self):
        """测试批量计算"""
        samples = [
            Sample(0, 0, np.array([1.0, 0.0]), np.array([1.0, 0.0]), 'honest', {}),
            Sample(1, 0, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 'noise', {}),
        ]
        results = CosineSimilarityCalculator.compute_batch(samples)

        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].cosine_similarity, 1.0, places=5)
        self.assertAlmostEqual(results[1].cosine_similarity, 0.0, places=5)


class TestSeparabilityAnalyzer(unittest.TestCase):
    """测试可分性分析器"""

    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.results = []

        # 诚实样本: 高相似度
        for i in range(50):
            self.results.append(
                SimilarityResult(i, 0, 'honest', np.random.uniform(0.95, 0.99), {})
            )

        # 噪声样本: 低相似度
        for i in range(50):
            self.results.append(
                SimilarityResult(i, 0, 'random_noise', np.random.uniform(-0.1, 0.1), {})
            )

        # 重放样本: 中等相似度
        for i in range(50):
            self.results.append(
                SimilarityResult(i, 0, 'replay', np.random.uniform(0.3, 0.6), {})
            )

        self.analyzer = SeparabilityAnalyzer(self.results)

    def test_statistics_computation(self):
        """测试统计计算"""
        stats = self.analyzer.compute_statistics()

        self.assertIn('honest', stats)
        self.assertIn('random_noise', stats)
        self.assertIn('replay', stats)

        # 诚实样本均值应该更高
        self.assertGreater(stats['honest']['mean'], stats['random_noise']['mean'])
        self.assertGreater(stats['honest']['mean'], stats['replay']['mean'])

    def test_separation_gap(self):
        """测试分离间隙计算"""
        gaps = self.analyzer.compute_separation_gap()

        self.assertIn('random_noise', gaps)
        self.assertIn('replay', gaps)

        # 噪声攻击应该有正的分离间隙
        self.assertGreater(gaps['random_noise'], 0)

    def test_roc_auc(self):
        """测试 ROC-AUC 计算"""
        auc_noise = self.analyzer.compute_roc_auc('random_noise')
        auc_replay = self.analyzer.compute_roc_auc('replay')

        # AUC 应该在 0.5 到 1.0 之间
        self.assertGreater(auc_noise, 0.9)  # 噪声应该很容易区分
        self.assertGreater(auc_replay, 0.5)
        self.assertLessEqual(auc_noise, 1.0)
        self.assertLessEqual(auc_replay, 1.0)

    def test_optimal_threshold(self):
        """测试最优阈值查找"""
        optimal = self.analyzer.find_optimal_threshold('random_noise', metric='f1')

        self.assertIn('optimal_threshold', optimal)
        self.assertIn('metrics_at_optimal', optimal)
        self.assertIn('F1', optimal['metrics_at_optimal'])

        # 最优阈值应该在 0 和 1 之间
        self.assertGreaterEqual(optimal['optimal_threshold'], 0)
        self.assertLessEqual(optimal['optimal_threshold'], 1)

    def test_roc_metrics(self):
        """测试 ROC 指标计算"""
        metrics = self.analyzer.compute_roc_metrics(threshold=0.5, attack_label='random_noise')

        self.assertIn('TPR', metrics)
        self.assertIn('FPR', metrics)
        self.assertIn('F1', metrics)

        # 指标应该在 0 和 1 之间
        self.assertGreaterEqual(metrics['TPR'], 0)
        self.assertLessEqual(metrics['TPR'], 1)


class TestAttackSampleGenerator(unittest.TestCase):
    """测试攻击样本生成器"""

    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.honest_samples = []

        for i in range(20):
            x_prev = np.random.randn(768).astype(np.float32)
            delta = np.random.randn(768).astype(np.float32) * 0.1
            x_curr = x_prev + delta

            self.honest_samples.append(
                Sample(i, 4, x_prev, x_curr, 'honest', {})
            )

        self.generator = AttackSampleGenerator(self.honest_samples, seed=42)

    def test_random_noise_generation(self):
        """测试随机噪声攻击生成"""
        noise_samples = self.generator.generate_random_noise([0.5, 1.0])

        # 每个诚实样本生成 2 个噪声样本
        self.assertEqual(len(noise_samples), len(self.honest_samples) * 2)

        # 检查噪声样本标签
        labels = set(s.label for s in noise_samples)
        self.assertEqual(labels, {'random_noise_sigma_0.5', 'random_noise_sigma_1.0'})

    def test_replay_attack_generation(self):
        """测试重放攻击生成"""
        replay_samples = self.generator.generate_replay_attacks(
            ['cross_sequence_same_layer'],
            pool_size=10
        )

        # 应该生成了一些样本
        self.assertGreater(len(replay_samples), 0)

        # 检查重放样本标签
        for s in replay_samples:
            self.assertTrue(s.label.startswith('replay_'))
            self.assertIn('replay_type', s.metadata)

    def test_layer_skipping_generation(self):
        """测试层跳过攻击生成"""
        skip_samples = self.generator.generate_layer_skipping()

        self.assertEqual(len(skip_samples), len(self.honest_samples))

        # 层跳过的 x_curr 应该等于 x_prev
        for i, sample in enumerate(skip_samples):
            np.testing.assert_array_almost_equal(sample.x_curr, sample.x_prev)


class TestResidualProperty(unittest.TestCase):
    """
    测试残差连接性质
    核心假设: 合法前向传播的余弦相似度应该较高
    """

    def test_residual_connection_similarity(self):
        """测试残差连接的相似度特性"""
        np.random.seed(42)

        # 模拟残差连接: x_k = x_{k-1} + delta
        x_prev = np.random.randn(768).astype(np.float32)

        # 情况1: 小的残差变化 (合法)
        delta_small = np.random.randn(768).astype(np.float32) * 0.1
        x_curr_legal = x_prev + delta_small
        sim_legal = CosineSimilarityCalculator.compute(x_prev, x_curr_legal)

        # 情况2: 随机噪声 (攻击)
        x_curr_noise = np.random.randn(768).astype(np.float32)
        sim_noise = CosineSimilarityCalculator.compute(x_prev, x_curr_noise)

        # 合法样本的相似度应该远高于噪声
        self.assertGreater(sim_legal, sim_noise)
        self.assertGreater(sim_legal, 0.5)  # 合法样本应该较高
        self.assertLess(sim_noise, 0.5)  # 噪声应该较低


class TestStage1ThreatModel(unittest.TestCase):
    """
    测试 Stage 1 威胁模型覆盖
    验证 Stage 1 能检测的两种攻击类型
    """

    def test_random_noise_detection(self):
        """测试随机噪声可被检测"""
        np.random.seed(42)

        # 生成大量测试数据
        honest_sims = []
        noise_sims = []

        for _ in range(100):
            x_prev = np.random.randn(768).astype(np.float32)

            # 诚实样本
            delta = np.random.randn(768).astype(np.float32) * 0.1
            x_curr = x_prev + delta
            honest_sims.append(CosineSimilarityCalculator.compute(x_prev, x_curr))

            # 噪声攻击
            x_noise = np.random.randn(768).astype(np.float32) * 2.0
            noise_sims.append(CosineSimilarityCalculator.compute(x_prev, x_noise))

        # 应该有明显的分离
        min_honest = np.min(honest_sims)
        max_noise = np.max(noise_sims)

        self.assertGreater(min_honest, max_noise,
                          "诚实样本和噪声样本应该有分离")

    def test_replay_attack_detection(self):
        """测试重放攻击可被检测"""
        np.random.seed(42)

        honest_sims = []
        replay_sims = []

        for _ in range(100):
            x_prev = np.random.randn(768).astype(np.float32)

            # 诚实样本
            delta = np.random.randn(768).astype(np.float32) * 0.1
            x_curr = x_prev + delta
            honest_sims.append(CosineSimilarityCalculator.compute(x_prev, x_curr))

            # 重放攻击 (不相关的向量)
            x_replay = np.random.randn(768).astype(np.float32)
            replay_sims.append(CosineSimilarityCalculator.compute(x_prev, x_replay))

        # 诚实样本的相似度应该更高
        mean_honest = np.mean(honest_sims)
        mean_replay = np.mean(replay_sims)

        self.assertGreater(mean_honest, mean_replay)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_zero_vector(self):
        """测试零向量处理"""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 2.0, 3.0])

        sim = CosineSimilarityCalculator.compute(v1, v2)
        self.assertEqual(sim, 0.0)  # 应该返回 0 而不是 NaN

    def test_very_small_vectors(self):
        """测试极小向量"""
        v1 = np.array([1e-10, 2e-10, 3e-10])
        v2 = np.array([1e-10, 2e-10, 3e-10])

        sim = CosineSimilarityCalculator.compute(v1, v2)
        # 极小向量可能因数值精度返回0，这是可接受的
        self.assertTrue(sim == 1.0 or sim == 0.0)

    def test_high_dimensional(self):
        """测试高维向量"""
        dim = 4096  # 大模型隐藏维度
        v1 = np.random.randn(dim).astype(np.float32)
        v2 = v1 + np.random.randn(dim).astype(np.float32) * 0.1

        sim = CosineSimilarityCalculator.compute(v1, v2)
        self.assertGreater(sim, 0.8)  # 高维下相似度应该仍然较高


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
