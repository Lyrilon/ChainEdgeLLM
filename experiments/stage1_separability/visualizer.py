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
        绘制分布直方图 - 使用双y轴解决密度差异问题

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
            'honest': '#2ecc71',
            'random_noise': '#e74c3c',
            'replay': '#3498db',
        }

        honest_data = data.get('honest', [])
        random_noise_labels = [k for k in data.keys() if k.startswith('random_noise')]
        replay_labels = [k for k in data.keys() if k.startswith('replay')]

        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # ========== 左图：honest + random_noise（双y轴）==========
        if random_noise_labels:
            # 主y轴：random_noise（红色）
            for label in sorted(random_noise_labels):
                ax1.hist(data[label], bins=50, alpha=0.6, color=colors['random_noise'],
                        label=label, density=True)
            ax1.set_ylabel('Density (Random Noise)', color=colors['random_noise'], fontsize=12)
            ax1.tick_params(axis='y', labelcolor=colors['random_noise'])
            ax1.set_ylim(bottom=0)

        # 次y轴：honest（绿色）- 显示在右侧
        if honest_data:
            ax1_twin = ax1.twinx()
            ax1_twin.hist(honest_data, bins=50, alpha=0.7, color=colors['honest'],
                         label=f'honest (n={len(honest_data)})', density=True)
            ax1_twin.set_ylabel('Density (Honest)', color=colors['honest'], fontsize=12)
            ax1_twin.tick_params(axis='y', labelcolor=colors['honest'])
            ax1_twin.set_ylim(bottom=0)

        ax1.set_xlabel('Cosine Similarity', fontsize=12)
        ax1.set_title('Honest vs Random Noise (Dual Y-axis)', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        if honest_data:
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')

        # ========== 右图：honest + replay（双y轴）==========
        if replay_labels:
            # 主y轴：replay（蓝色）
            for label in sorted(replay_labels):
                ax2.hist(data[label], bins=50, alpha=0.6, color=colors['replay'],
                        label=label, density=True)
            ax2.set_ylabel('Density (Replay)', color=colors['replay'], fontsize=12)
            ax2.tick_params(axis='y', labelcolor=colors['replay'])
            ax2.set_ylim(bottom=0)

        # 次y轴：honest（绿色）
        if honest_data:
            ax2_twin = ax2.twinx()
            ax2_twin.hist(honest_data, bins=50, alpha=0.7, color=colors['honest'],
                         label=f'honest (n={len(honest_data)})', density=True)
            ax2_twin.set_ylabel('Density (Honest)', color=colors['honest'], fontsize=12)
            ax2_twin.tick_params(axis='y', labelcolor=colors['honest'])
            ax2_twin.set_ylim(bottom=0)

        ax2.set_xlabel('Cosine Similarity', fontsize=12)
        ax2.set_title('Honest vs Replay (Dual Y-axis)', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # x轴范围自适应
        all_replay_data = [v for l in replay_labels for v in data[l]] + honest_data
        if all_replay_data:
            x_min, x_max = min(all_replay_data), max(all_replay_data)
            margin = (x_max - x_min) * 0.1
            ax2.set_xlim(x_min - margin, x_max + margin)

        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        if honest_data:
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax2.legend(loc='upper left')

        title = f'Distribution of Cosine Similarities'
        if layer_idx is not None:
            title += f' (Layer {layer_idx})'
        fig.suptitle(title, fontsize=16, fontweight='bold')

        name = f"distribution_histogram_layer{layer_idx if layer_idx else 'all'}"
        self._save_figure(name)

    def plot_box_violin(self, results: List, layer_idx: int = None):
        """
        绘制箱线图和小提琴图 - 分开画图：honest+random_noise 和 honest+replay 各一张

        Args:
            results: SimilarityResult 列表
            layer_idx: 特定层索引
        """
        # 按标签分组
        data = defaultdict(list)
        for r in results:
            if layer_idx is None or r.layer_idx == layer_idx:
                data[r.label].append(r.cosine_similarity)

        # 分类标签
        honest_data = data.get('honest', [])
        random_noise_labels = [k for k in data.keys() if k.startswith('random_noise')]
        replay_labels = [k for k in data.keys() if k.startswith('replay')]

        colors = {
            'honest': '#2ecc71',
            'random_noise': '#e74c3c',
            'replay': '#3498db',
        }

        # ========== 图1: honest + random_noise 箱型图 ==========
        if honest_data and random_noise_labels:
            fig, ax = plt.subplots(figsize=(10, 6))

            labels = ['honest'] + sorted(random_noise_labels)
            values = [data[l] for l in labels]
            label_colors = [colors.get(l.split('_')[0], '#95a5a6') for l in labels]

            bp = ax.boxplot(values, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], label_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel('Cosine Similarity', fontsize=12)
            ax.set_title('Box Plot: Honest vs Random Noise', fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            name = f"box_honest_vs_noise_layer{layer_idx if layer_idx else 'all'}"
            self._save_figure(name)

        # ========== 图2: honest + replay 箱型图 ==========
        if honest_data and replay_labels:
            fig, ax = plt.subplots(figsize=(10, 6))

            labels = ['honest'] + sorted(replay_labels)
            values = [data[l] for l in labels]
            label_colors = [colors.get(l.split('_')[0], '#95a5a6') for l in labels]

            bp = ax.boxplot(values, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], label_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel('Cosine Similarity', fontsize=12)
            ax.set_title('Box Plot: Honest vs Replay', fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # 设置 y 轴范围以更好显示 honest 和 replay 的差异
            all_values = [v for sublist in values for v in sublist]
            if all_values:
                y_min, y_max = min(all_values), max(all_values)
                margin = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - margin, y_max + margin)

            name = f"box_honest_vs_replay_layer{layer_idx if layer_idx else 'all'}"
            self._save_figure(name)

        # ========== 图3: honest + random_noise 小提琴图 ==========
        if honest_data and random_noise_labels:
            fig, ax = plt.subplots(figsize=(10, 6))

            labels = ['honest'] + sorted(random_noise_labels)
            values = [data[l] for l in labels]

            parts = ax.violinplot(values, showmeans=True, showmedians=True)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_ylabel('Cosine Similarity', fontsize=12)
            ax.set_title('Violin Plot: Honest vs Random Noise', fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            name = f"violin_honest_vs_noise_layer{layer_idx if layer_idx else 'all'}"
            self._save_figure(name)

        # ========== 图4: honest + replay 小提琴图 ==========
        if honest_data and replay_labels:
            fig, ax = plt.subplots(figsize=(10, 6))

            labels = ['honest'] + sorted(replay_labels)
            values = [data[l] for l in labels]

            parts = ax.violinplot(values, showmeans=True, showmedians=True)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_ylabel('Cosine Similarity', fontsize=12)
            ax.set_title('Violin Plot: Honest vs Replay', fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # 设置 y 轴范围以更好显示 honest 和 replay 的差异
            all_values = [v for sublist in values for v in sublist]
            if all_values:
                y_min, y_max = min(all_values), max(all_values)
                margin = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - margin, y_max + margin)

            name = f"violin_honest_vs_replay_layer{layer_idx if layer_idx else 'all'}"
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
        比较不同层的性能 - 分开画图：honest+random_noise 和 honest+replay

        Args:
            results: SimilarityResult 列表
        """
        # 按层和标签分组
        data = defaultdict(lambda: defaultdict(list))
        for r in results:
            data[r.layer_idx][r.label].append(r.cosine_similarity)

        layers = sorted(data.keys())
        honest_label = 'honest'
        random_noise_labels = sorted([l for l in data[layers[0]].keys() if l.startswith('random_noise')])
        replay_labels = sorted([l for l in data[layers[0]].keys() if l.startswith('replay')])

        colors = {
            'honest': '#2ecc71',
            'random_noise': '#e74c3c',
            'replay': '#3498db',
        }

        # ========== 图1: honest + random_noise ==========
        if honest_label and random_noise_labels:
            fig, ax = plt.subplots(figsize=(12, 6))

            labels_to_plot = [honest_label] + random_noise_labels
            x = np.arange(len(layers))
            width = 0.8 / len(labels_to_plot)

            for i, label in enumerate(labels_to_plot):
                means = []
                stds = []
                for layer in layers:
                    scores = data[layer].get(label, [])
                    means.append(np.mean(scores) if scores else 0)
                    stds.append(np.std(scores) if scores else 0)

                offset = (i - len(labels_to_plot) / 2) * width + width / 2
                color = colors.get(label.split('_')[0], '#95a5a6')
                ax.bar(x + offset, means, width, yerr=stds, label=label, color=color, alpha=0.8)

            ax.set_xlabel('Layer Index', fontsize=12)
            ax.set_ylabel('Cosine Similarity', fontsize=12)
            ax.set_title('Mean Cosine Similarity by Layer: Honest vs Random Noise', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(layers)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            self._save_figure("layer_comparison_honest_vs_noise")

        # ========== 图2: honest + replay ==========
        if honest_label and replay_labels:
            fig, ax = plt.subplots(figsize=(12, 6))

            labels_to_plot = [honest_label] + replay_labels
            x = np.arange(len(layers))
            width = 0.8 / len(labels_to_plot)

            for i, label in enumerate(labels_to_plot):
                means = []
                stds = []
                for layer in layers:
                    scores = data[layer].get(label, [])
                    means.append(np.mean(scores) if scores else 0)
                    stds.append(np.std(scores) if scores else 0)

                offset = (i - len(labels_to_plot) / 2) * width + width / 2
                color = colors.get(label.split('_')[0], '#95a5a6')
                ax.bar(x + offset, means, width, yerr=stds, label=label, color=color, alpha=0.8)

            ax.set_xlabel('Layer Index', fontsize=12)
            ax.set_ylabel('Cosine Similarity', fontsize=12)
            ax.set_title('Mean Cosine Similarity by Layer: Honest vs Replay', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(layers)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            # 自适应 y 轴范围
            all_means = []
            for label in labels_to_plot:
                for layer in layers:
                    scores = data[layer].get(label, [])
                    if scores:
                        all_means.append(np.mean(scores))
            if all_means:
                y_min, y_max = min(all_means), max(all_means)
                margin = (y_max - y_min) * 0.2
                ax.set_ylim(y_min - margin, y_max + margin)

            self._save_figure("layer_comparison_honest_vs_replay")

    def plot_confusion_matrices(self, analyzer, attack_labels: List[str]):
        """
        绘制混淆矩阵热力图 - 每个攻击类型单独保存到文件夹

        Args:
            analyzer: SeparabilityAnalyzer 实例
            attack_labels: 攻击类型列表
        """
        import matplotlib.patches as mpatches

        # 创建混淆矩阵子文件夹
        cm_dir = os.path.join(self.output_dir, "confusion_matrices")
        os.makedirs(cm_dir, exist_ok=True)

        for attack_label in attack_labels:
            if attack_label not in analyzer.attack_scores:
                continue

            # 获取混淆矩阵
            cm_result = analyzer.compute_confusion_matrix(attack_label)
            if not cm_result:
                continue

            cm = np.array([
                [cm_result['confusion_matrix']['TP'], cm_result['confusion_matrix']['FN']],
                [cm_result['confusion_matrix']['FP'], cm_result['confusion_matrix']['TN']]
            ])

            # 创建单独的图片
            fig, ax = plt.subplots(figsize=(8, 6))

            # 绘制热力图
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)

            # 设置刻度和标签
            classes = ['Honest', 'Attack']
            ax.set(xticks=[0, 1],
                   yticks=[0, 1],
                   xticklabels=classes,
                   yticklabels=classes,
                   title=f'{attack_label}\n(Threshold: {cm_result["threshold"]:.4f})',
                   ylabel='True label',
                   xlabel='Predicted label')

            # 在每个单元格中显示数值
            thresh = cm.max() / 2.
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=24, fontweight='bold')

            # 添加指标文本框
            metrics_text = (
                f"Accuracy:    {cm_result['metrics']['accuracy']:.4f}\n"
                f"Precision:   {cm_result['metrics']['precision']:.4f}\n"
                f"Recall:      {cm_result['metrics']['recall']:.4f}\n"
                f"F1 Score:    {cm_result['metrics']['f1_score']:.4f}\n"
                f"Specificity: {cm_result['metrics']['specificity']:.4f}\n"
                f"FPR:         {cm_result['metrics']['fpr']:.6f}\n"
                f"FNR:         {cm_result['metrics']['fnr']:.6f}"
            )
            ax.text(1.25, 0.5, metrics_text,
                   transform=ax.transAxes,
                   fontsize=11, verticalalignment='center',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # 调整布局以容纳右侧文本
            plt.tight_layout()
            plt.subplots_adjust(right=0.75)

            # 保存到子文件夹
            safe_label = attack_label.replace(' ', '_').replace('/', '_')
            save_path = os.path.join(cm_dir, f"cm_{safe_label}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"保存混淆矩阵: {save_path}")
            plt.close()

        print(f"所有混淆矩阵已保存到: {cm_dir}/")

    def save_detailed_report(self, analyzer, attack_labels: List[str], output_file: str = "detailed_report.txt"):
        """
        保存详细的文本报告

        Args:
            analyzer: SeparabilityAnalyzer 实例
            attack_labels: 攻击类型列表
            output_file: 输出文件名
        """
        report = analyzer.generate_detailed_report()

        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"详细报告已保存: {output_path}")

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
