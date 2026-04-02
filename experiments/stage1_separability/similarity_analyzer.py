"""
Similarity Analyzer Module
计算余弦相似度并分析可分性
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """相似度分析结果"""
    sample_id: int
    layer_idx: int
    label: str
    cosine_similarity: float
    metadata: Dict


class CosineSimilarityCalculator:
    """
    余弦相似度计算器
    实现论文公式 (1): S(x_{k-1}, x_k) = (x_{k-1} · x_k) / (||x_{k-1}|| ||x_k||)
    """

    @staticmethod
    def compute(x_prev: np.ndarray, x_curr: np.ndarray, eps: float = 1e-8) -> float:
        """
        计算两个向量的余弦相似度

        Args:
            x_prev: 前一隐藏状态
            x_curr: 当前隐藏状态
            eps: 数值稳定性常数

        Returns:
            余弦相似度 [-1, 1]
        """
        # 展平向量
        x_prev_flat = x_prev.flatten()
        x_curr_flat = x_curr.flatten()

        # 计算点积
        dot_product = np.dot(x_prev_flat, x_curr_flat)

        # 计算范数
        norm_prev = np.linalg.norm(x_prev_flat)
        norm_curr = np.linalg.norm(x_curr_flat)

        # 避免除零
        if norm_prev < eps or norm_curr < eps:
            return 0.0

        # 计算余弦相似度
        cosine_sim = dot_product / (norm_prev * norm_curr + eps)

        # 裁剪到有效范围
        return float(np.clip(cosine_sim, -1.0, 1.0))

    @staticmethod
    def compute_batch(samples: List) -> List[SimilarityResult]:
        """
        批量计算相似度

        Args:
            samples: Sample 对象列表

        Returns:
            SimilarityResult 列表
        """
        results = []
        for sample in samples:
            sim = CosineSimilarityCalculator.compute(sample.x_prev, sample.x_curr)
            result = SimilarityResult(
                sample_id=sample.sample_id,
                layer_idx=sample.layer_idx,
                label=sample.label,
                cosine_similarity=sim,
                metadata=sample.metadata
            )
            results.append(result)
        return results


class SeparabilityAnalyzer:
    """
    可分性分析器
    分析诚实样本和攻击样本的分布可分性
    """

    def __init__(self, results: List[SimilarityResult]):
        self.results = results
        self.honest_scores = None
        self.attack_scores = defaultdict(list)
        self._organize_data()

    def _organize_data(self):
        """组织数据按标签分组"""
        for result in self.results:
            if result.label == 'honest':
                if self.honest_scores is None:
                    self.honest_scores = []
                self.honest_scores.append(result.cosine_similarity)
            else:
                self.attack_scores[result.label].append(result.cosine_similarity)

        if self.honest_scores is not None:
            self.honest_scores = np.array(self.honest_scores)

        for key in self.attack_scores:
            self.attack_scores[key] = np.array(self.attack_scores[key])

    def compute_statistics(self) -> Dict:
        """
        计算统计指标

        Returns:
            统计结果字典
        """
        stats_dict = {}

        # 诚实样本统计
        if self.honest_scores is not None and len(self.honest_scores) > 0:
            stats_dict['honest'] = {
                'count': len(self.honest_scores),
                'mean': float(np.mean(self.honest_scores)),
                'std': float(np.std(self.honest_scores)),
                'min': float(np.min(self.honest_scores)),
                'max': float(np.max(self.honest_scores)),
                'median': float(np.median(self.honest_scores)),
                'q25': float(np.percentile(self.honest_scores, 25)),
                'q75': float(np.percentile(self.honest_scores, 75)),
            }

        # 攻击样本统计
        for label, scores in self.attack_scores.items():
            if len(scores) > 0:
                stats_dict[label] = {
                    'count': len(scores),
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'median': float(np.median(scores)),
                    'q25': float(np.percentile(scores, 25)),
                    'q75': float(np.percentile(scores, 75)),
                }

        return stats_dict

    def compute_separation_gap(self) -> Dict[str, float]:
        """
        计算分离间隙 (Separation Gap)
        Δ = min(S_honest) - max(S_malicious)

        Returns:
            分离间隙字典
        """
        if self.honest_scores is None or len(self.honest_scores) == 0:
            return {}

        min_honest = np.min(self.honest_scores)

        gaps = {}
        for label, scores in self.attack_scores.items():
            if len(scores) > 0:
                max_attack = np.max(scores)
                gap = min_honest - max_attack
                gaps[label] = float(gap)

        return gaps

    def compute_roc_metrics(self, threshold: float, attack_label: str) -> Dict:
        """
        计算 ROC 指标

        Args:
            threshold: 分类阈值
            attack_label: 攻击类型标签

        Returns:
            指标字典 (TPR, FPR, Precision, Recall, F1)
        """
        if self.honest_scores is None or attack_label not in self.attack_scores:
            return {}

        attack_scores = self.attack_scores[attack_label]

        # 二分类: honest = 正类 (1), attack = 负类 (0)
        # 在我们的场景中，低于阈值的是攻击
        y_true = np.concatenate([
            np.ones(len(self.honest_scores)),
            np.zeros(len(attack_scores))
        ])
        y_scores = np.concatenate([
            self.honest_scores,
            attack_scores
        ])
        y_pred = (y_scores >= threshold).astype(int)

        # 计算混淆矩阵
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # 计算指标
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0

        return {
            'threshold': threshold,
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn),
            'TPR': float(tpr),
            'FPR': float(fpr),
            'Precision': float(precision),
            'Recall': float(tpr),
            'F1': float(f1),
        }

    def find_optimal_threshold(self, attack_label: str, metric: str = 'f1') -> Dict:
        """
        寻找最优阈值

        Args:
            attack_label: 攻击类型
            metric: 优化指标 (f1, accuracy, youden)

        Returns:
            最优阈值及对应指标
        """
        if self.honest_scores is None or attack_label not in self.attack_scores:
            return {}

        attack_scores = self.attack_scores[attack_label]

        # 确定搜索范围
        all_scores = np.concatenate([self.honest_scores, attack_scores])
        min_score, max_score = np.min(all_scores), np.max(all_scores)

        # 搜索阈值
        thresholds = np.linspace(min_score, max_score, 100)
        best_threshold = 0.5
        best_value = -1

        results = []
        for threshold in thresholds:
            metrics = self.compute_roc_metrics(threshold, attack_label)
            results.append((threshold, metrics))

            if metric == 'f1':
                value = metrics.get('F1', 0)
            elif metric == 'accuracy':
                value = (metrics.get('TP', 0) + metrics.get('TN', 0)) / \
                        (metrics.get('TP', 0) + metrics.get('TN', 0) +
                         metrics.get('FP', 0) + metrics.get('FN', 0) + 1e-8)
            elif metric == 'youden':
                value = metrics.get('TPR', 0) - metrics.get('FPR', 0)
            else:
                value = metrics.get('F1', 0)

            if value > best_value:
                best_value = value
                best_threshold = threshold

        return {
            'optimal_threshold': float(best_threshold),
            'best_metric_value': float(best_value),
            'optimized_for': metric,
            'metrics_at_optimal': self.compute_roc_metrics(best_threshold, attack_label)
        }

    def compute_roc_auc(self, attack_label: str) -> float:
        """
        计算 ROC-AUC

        Args:
            attack_label: 攻击类型

        Returns:
            ROC-AUC 值
        """
        if self.honest_scores is None or attack_label not in self.attack_scores:
            return 0.0

        attack_scores = self.attack_scores[attack_label]

        # 合并数据
        y_true = np.concatenate([
            np.ones(len(self.honest_scores)),
            np.zeros(len(attack_scores))
        ])
        y_scores = np.concatenate([
            self.honest_scores,
            attack_scores
        ])

        # 计算 AUC
        try:
            auc = self._auc_trapz(y_true, y_scores)
        except:
            auc = 0.5

        return float(auc)

    @staticmethod
    def _auc_trapz(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """使用梯形法则计算 AUC"""
        # 按分数排序
        order = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[order]

        # 计算 TPR 和 FPR
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos

        tpr_list = [0.0]
        fpr_list = [0.0]

        tp = 0
        fp = 0
        for label in y_true_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        # 计算曲线下面积
        auc = np.trapz(tpr_list, fpr_list)
        return auc

    def kolmogorov_smirnov_test(self, attack_label: str) -> Dict:
        """
        Kolmogorov-Smirnov 检验
        测试诚实样本和攻击样本是否来自同一分布

        Args:
            attack_label: 攻击类型

        Returns:
            KS 检验结果
        """
        if self.honest_scores is None or attack_label not in self.attack_scores:
            return {}

        attack_scores = self.attack_scores[attack_label]

        statistic, p_value = stats.ks_2samp(self.honest_scores, attack_scores)

        return {
            'ks_statistic': float(statistic),
            'p_value': float(p_value),
            'significant_at_0.05': bool(p_value < 0.05),
            'significant_at_0.01': bool(p_value < 0.01),
        }

    def comprehensive_analysis(self) -> Dict:
        """
        综合分析

        Returns:
            完整分析报告
        """
        analysis = {
            'statistics': self.compute_statistics(),
            'separation_gaps': self.compute_separation_gap(),
            'optimal_thresholds': {},
            'roc_auc': {},
            'ks_tests': {},
        }

        for attack_label in self.attack_scores.keys():
            analysis['optimal_thresholds'][attack_label] = self.find_optimal_threshold(attack_label)
            analysis['roc_auc'][attack_label] = self.compute_roc_auc(attack_label)
            analysis['ks_tests'][attack_label] = self.kolmogorov_smirnov_test(attack_label)

        return analysis


def test_similarity_analyzer():
    """测试相似度分析器"""
    # 创建模拟数据
    np.random.seed(42)

    # 诚实样本: 高相似度 (0.95-0.99)
    honest_sims = np.random.uniform(0.95, 0.99, 100)

    # 随机噪声: 低相似度 (-0.1-0.1)
    noise_sims = np.random.uniform(-0.1, 0.1, 100)

    # 重放攻击: 中等相似度 (0.3-0.6)
    replay_sims = np.random.uniform(0.3, 0.6, 100)

    # 构建结果
    results = []
    for i, sim in enumerate(honest_sims):
        results.append(SimilarityResult(i, 4, 'honest', sim, {}))
    for i, sim in enumerate(noise_sims):
        results.append(SimilarityResult(i, 4, 'random_noise', sim, {}))
    for i, sim in enumerate(replay_sims):
        results.append(SimilarityResult(i, 4, 'replay', sim, {}))

    # 分析
    analyzer = SeparabilityAnalyzer(results)

    print("统计结果:")
    print(analyzer.compute_statistics())

    print("\n分离间隙:")
    print(analyzer.compute_separation_gap())

    print("\n最优阈值 (random_noise):")
    print(analyzer.find_optimal_threshold('random_noise'))

    print("\nROC-AUC:")
    print(f"random_noise: {analyzer.compute_roc_auc('random_noise'):.4f}")
    print(f"replay: {analyzer.compute_roc_auc('replay'):.4f}")


if __name__ == "__main__":
    test_similarity_analyzer()
