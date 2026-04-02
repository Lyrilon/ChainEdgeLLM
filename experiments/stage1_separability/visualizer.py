"""
Visualizer Module
可视化 Stage 1 可分性实验结果
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import os
import json
from collections import defaultdict

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
sns.set_palette("husl")


class ResultVisualizer:
    """
    结果可视化器
    """

    def __init__(self, output_dir: str = "./results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save_figure(self, name: str, dpi: int = 300):
        """保存图片"""
        path = os.path.join(self.output_dir, f"{name}.png")
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f"保存图片: {path}")
        plt.close()

    def plot_distribution_histogram(self, results: List, layer_idx: int = None):
        """
        绘制分布直方图 - 使用双轴展示所有数据

        Args:
            results: SimilarityResult 列表
            layer_idx: 特定层索引 (None 表示所有层)
        """
        # 按标签分组
        data = defaultdict(list)
        for r in results:
            if layer_idx is None or r.layer_idx == layer_idx:
                data[r.label].append(r.cosine_similarity)

        colors = {
            'honest': '#2ecc71',  # 绿色
            'random_noise': '#e74c3c',  # 红色
            'replay': '#3498db',  # 蓝色
            'layer_skip': '#f39c12',  # 橙色
            'precision_downgrade': '#9b59b6',  # 紫色
        }

        # 创建两个子图：左图展示完整范围，右图展示诚实样本细节
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 左图：完整范围（展示攻击样本）
        for label, scores in data.items():
            color = colors.get(label.split('_')[0], None)
            ax1.hist(scores, bins=50, alpha=0.6, label=label, color=color, density=True)

        ax1.set_xlabel('Cosine Similarity (Full Range)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Full Distribution View', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.2, 1.1)  # 固定范围以展示所有数据

        # 右图：诚实样本细节（放大视图）
        if 'honest' in data:
            honest_scores = data['honest']
            ax2.hist(honest_scores, bins=50, alpha=0.7, color='#2ecc71',
                     label=f'honest (n={len(honest_scores)})', density=True)
            ax2.set_xlabel('Cosine Similarity (Honest Detail)', fontsize=12)
            ax2.set_ylabel('Density', fontsize=12)

            # 自适应范围：均值 ± 5倍标准差
            mean_val = np.mean(honest_scores)
            std_val = np.std(honest_scores)
            ax2.set_xlim(max(-0.1, mean_val - 5*std_val), min(1.05, mean_val + 5*std_val))
            ax2.set_title(f'Honest Samples Detail (μ={mean_val:.4f}, σ={std_val:.4f})', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        title = f'Distribution of Cosine Similarities'
        if layer_idx is not None:
            title += f' (Layer {layer_idx})'
        fig.suptitle(title, fontsize=16, fontweight='bold')

        name = f"distribution_histogram_layer{layer_idx if layer_idx else 'all'}"
        self._save_figure(name)

    def plot_box_violin(self, results: List, layer_idx: int = None):
        """
        绘制箱线图和小提琴图

        Args:
            results: SimilarityResult 列表
            layer_idx: 特定层索引
        """
        # 按标签分组
        data = defaultdict(list)
        for r in results:
            if layer_idx is None or r.layer_idx == layer_idx:
                data[r.label].append(r.cosine_similarity)

        # 准备数据
        labels = []
        values = []
        for label in sorted(data.keys(), key=lambda x: (x != 'honest', x)):
            labels.append(label)
            values.append(data[label])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 箱线图
        bp = ax1.boxplot(values, labels=labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax1.set_ylabel('Cosine Similarity', fontsize=12)
        ax1.set_title('Box Plot', fontsize=14)
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 小提琴图
        parts = ax2.violinplot(values, showmeans=True, showmedians=True)
        ax2.set_xticks(range(1, len(labels) + 1))
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Cosine Similarity', fontsize=12)
        ax2.set_title('Violin Plot', fontsize=14)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        name = f"box_violin_layer{layer_idx if layer_idx else 'all'}"
        self._save_figure(name)

    def plot_roc_curve(self, analyzer, attack_labels: List[str]):
        """
        绘制 ROC 曲线

        Args:
            analyzer: SeparabilityAnalyzer 实例
            attack_labels: 攻击类型列表
        """
        plt.figure(figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(attack_labels)))

        for attack_label, color in zip(attack_labels, colors):
            if attack_label not in analyzer.attack_scores:
                continue

            attack_scores = analyzer.attack_scores[attack_label]

            # 构建 ROC 数据
            y_true = np.concatenate([
                np.ones(len(analyzer.honest_scores)),
                np.zeros(len(attack_scores))
            ])
            y_scores = np.concatenate([
                analyzer.honest_scores,
                attack_scores
            ])

            # 计算 ROC 点
            thresholds = np.linspace(0, 1, 100)
            tprs = []
            fprs = []

            for threshold in thresholds:
                y_pred = (y_scores >= threshold).astype(int)
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fn = np.sum((y_true == 1) & (y_pred == 0))

                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                tprs.append(tpr)
                fprs.append(fpr)

            auc = analyzer.compute_roc_auc(attack_label)
            plt.plot(fprs, tprs, label=f'{attack_label} (AUC={auc:.4f})', color=color, linewidth=2)

        # 对角线
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')

        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        self._save_figure("roc_curves")

    def plot_pr_curve(self, analyzer, attack_labels: List[str]):
        """
        绘制 Precision-Recall 曲线

        Args:
            analyzer: SeparabilityAnalyzer 实例
            attack_labels: 攻击类型列表
        """
        plt.figure(figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(attack_labels)))

        for attack_label, color in zip(attack_labels, colors):
            if attack_label not in analyzer.attack_scores:
                continue

            attack_scores = analyzer.attack_scores[attack_label]

            # 构建 PR 数据
            y_true = np.concatenate([
                np.ones(len(analyzer.honest_scores)),
                np.zeros(len(attack_scores))
            ])
            y_scores = np.concatenate([
                analyzer.honest_scores,
                attack_scores
            ])

            # 计算 PR 点
            thresholds = np.linspace(0, 1, 100)
            precisions = []
            recalls = []

            for threshold in thresholds:
                y_pred = (y_scores >= threshold).astype(int)
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                precisions.append(precision)
                recalls.append(recall)

            plt.plot(recalls, precisions, label=attack_label, color=color, linewidth=2)

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        self._save_figure("pr_curves")

    def plot_scatter_2d(self, results: List, layer_idx: int = None):
        """
        绘制散点图 (使用 PCA 降维)

        Args:
            results: SimilarityResult 列表 (需要包含原始向量)
            layer_idx: 特定层索引
        """
        from sklearn.decomposition import PCA

        # 准备数据
        vectors = []
        labels = []
        colors_map = []

        color_dict = {
            'honest': '#2ecc71',
            'random_noise': '#e74c3c',
            'replay': '#3498db',
            'layer_skip': '#f39c12',
        }

        for r in results:
            if layer_idx is None or r.layer_idx == layer_idx:
                if 'x_prev' in r.metadata and 'x_curr' in r.metadata:
                    # 连接前后状态作为特征
                    vec = np.concatenate([r.metadata['x_prev'], r.metadata['x_curr']])
                    vectors.append(vec)
                    labels.append(r.label)
                    colors_map.append(color_dict.get(r.label.split('_')[0], '#95a5a6'))

        if len(vectors) == 0:
            print("没有足够的数据绘制散点图")
            return

        # PCA 降维
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(np.array(vectors))

        plt.figure(figsize=(12, 8))

        # 按标签分组绘制
        unique_labels = list(set(labels))
        for label in unique_labels:
            mask = [l == label for l in labels]
            x = vectors_2d[mask, 0]
            y = vectors_2d[mask, 1]
            color = color_dict.get(label.split('_')[0], '#95a5a6')
            plt.scatter(x, y, label=label, alpha=0.6, s=50, color=color)

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        plt.title('PCA Projection of Hidden States', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        name = f"scatter_2d_layer{layer_idx if layer_idx else 'all'}"
        self._save_figure(name)

    def plot_threshold_sweep(self, analyzer, attack_labels: List[str]):
        """
        绘制阈值扫描图 (F1, Precision, Recall vs Threshold)

        Args:
            analyzer: SeparabilityAnalyzer 实例
            attack_labels: 攻击类型列表
        """
        fig, axes = plt.subplots(len(attack_labels), 1, figsize=(12, 4 * len(attack_labels)))
        if len(attack_labels) == 1:
            axes = [axes]

        for ax, attack_label in zip(axes, attack_labels):
            if attack_label not in analyzer.attack_scores:
                continue

            attack_scores = analyzer.attack_scores[attack_label]

            # 构建数据
            y_true = np.concatenate([
                np.ones(len(analyzer.honest_scores)),
                np.zeros(len(attack_scores))
            ])
            y_scores = np.concatenate([
                analyzer.honest_scores,
                attack_scores
            ])

            # 扫描阈值
            thresholds = np.linspace(0, 1, 100)
            f1s = []
            precs = []
            recs = []

            for threshold in thresholds:
                y_pred = (y_scores >= threshold).astype(int)
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                f1s.append(f1)
                precs.append(precision)
                recs.append(recall)

            ax.plot(thresholds, f1s, label='F1', linewidth=2)
            ax.plot(thresholds, precs, label='Precision', linewidth=2)
            ax.plot(thresholds, recs, label='Recall', linewidth=2)
            ax.set_xlabel('Threshold', fontsize=10)
            ax.set_ylabel('Score', fontsize=10)
            ax.set_title(f'{attack_label}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        plt.tight_layout()
        self._save_figure("threshold_sweep")

    def plot_layer_comparison(self, results: List):
        """
        比较不同层的性能

        Args:
            results: SimilarityResult 列表
        """
        # 按层和标签分组
        data = defaultdict(lambda: defaultdict(list))
        for r in results:
            data[r.layer_idx][r.label].append(r.cosine_similarity)

        layers = sorted(data.keys())
        labels = set()
        for layer_data in data.values():
            labels.update(layer_data.keys())
        labels = sorted(labels, key=lambda x: (x != 'honest', x))

        # 准备绘图数据
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(layers))
        width = 0.8 / len(labels)

        for i, label in enumerate(labels):
            means = []
            stds = []
            for layer in layers:
                scores = data[layer].get(label, [])
                means.append(np.mean(scores) if scores else 0)
                stds.append(np.std(scores) if scores else 0)

            offset = (i - len(labels) / 2) * width + width / 2
            ax.bar(x + offset, means, width, yerr=stds, label=label, alpha=0.8)

        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title('Mean Cosine Similarity by Layer', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        self._save_figure("layer_comparison")

    def generate_report(self, analysis_results: Dict, output_file: str = "report.json"):
        """
        生成分析报告

        Args:
            analysis_results: 分析结果字典
            output_file: 输出文件名
        """
        output_path = os.path.join(self.output_dir, output_file)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)

        print(f"报告已保存: {output_path}")

    def plot_all(self, results: List, analyzer, attack_labels: List[str]):
        """
        生成所有可视化

        Args:
            results: SimilarityResult 列表
            analyzer: SeparabilityAnalyzer 实例
            attack_labels: 攻击类型列表
        """
        print("生成可视化...")

        # 1. 分布直方图
        self.plot_distribution_histogram(results)

        # 2. 箱线图和小提琴图
        self.plot_box_violin(results)

        # 3. ROC 曲线
        self.plot_roc_curve(analyzer, attack_labels)

        # 4. PR 曲线
        self.plot_pr_curve(analyzer, attack_labels)

        # 5. 阈值扫描
        self.plot_threshold_sweep(analyzer, attack_labels)

        # 6. 层比较
        self.plot_layer_comparison(results)

        print("所有可视化已生成!")


if __name__ == "__main__":
    # 测试可视化器
    from similarity_analyzer import SimilarityResult, SeparabilityAnalyzer

    # 创建模拟数据
    np.random.seed(42)
    results = []

    for i in range(100):
        results.append(SimilarityResult(i, 4, 'honest', np.random.uniform(0.95, 0.99), {}))
        results.append(SimilarityResult(i, 4, 'random_noise', np.random.uniform(-0.1, 0.1), {}))
        results.append(SimilarityResult(i, 4, 'replay', np.random.uniform(0.3, 0.6), {}))

    # 分析
    analyzer = SeparabilityAnalyzer(results)

    # 可视化
    vis = ResultVisualizer()
    vis.plot_all(results, analyzer, ['random_noise', 'replay'])
