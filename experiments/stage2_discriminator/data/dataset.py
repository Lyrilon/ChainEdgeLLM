"""
PyTorch Dataset for Stage 2 Discriminator Training
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../stage1_separability'))

from data_generator import Sample


class DiscriminatorDataset(Dataset):
    """判别器训练数据集"""

    def __init__(self, samples: List[Sample], augment: bool = False,
                 aug_noise_std: float = 0.01, aug_dropout_p: float = 0.05,
                 normalize: bool = False, norm_stats: Optional[dict] = None):
        """
        normalize: 对 x_curr 和 delta 分别做 z-score 归一化（layer-specific）
        norm_stats: {'x_curr_mean', 'x_curr_std', 'delta_mean', 'delta_std'}
                    若为 None 则从当前数据集计算（仅用于训练集）
        """
        self.samples = samples
        self.augment = augment
        self.aug_noise_std = aug_noise_std
        self.aug_dropout_p = aug_dropout_p
        self.normalize = normalize

        # 提取特征和标签
        self.features = []
        self.labels = []
        self.attack_types = []
        self.layer_indices = []

        for sample in samples:
            # 拼接 x_curr 和差分 delta = x_curr - x_prev
            # 量化误差在 delta 空间中更显著（见 Stage 2 设计分析）
            delta = sample.x_curr - sample.x_prev
            feature = np.concatenate([sample.x_curr, delta])
            self.features.append(feature)
            # 二分类：honest=1, attack=0
            label = 1 if sample.label == 'honest' else 0
            self.labels.append(label)
            # 保存攻击类型（诚实样本为 'honest'）
            attack_type = sample.metadata.get('attack_type', 'honest') if sample.metadata else 'honest'
            self.attack_types.append(attack_type)
            self.layer_indices.append(sample.layer_idx)

        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        half = self.features.shape[1] // 2

        # 特征归一化：对 x_curr 和 delta 分别 z-score（layer-specific）
        if normalize:
            if norm_stats is None:
                self.norm_stats = {
                    'x_curr_mean': self.features[:, :half].mean(axis=0),
                    'x_curr_std':  self.features[:, :half].std(axis=0).clip(min=1e-6),
                    'delta_mean':  self.features[:, half:].mean(axis=0),
                    'delta_std':   self.features[:, half:].std(axis=0).clip(min=1e-6),
                }
            else:
                self.norm_stats = norm_stats
            self.features[:, :half] = (self.features[:, :half] - self.norm_stats['x_curr_mean']) / self.norm_stats['x_curr_std']
            self.features[:, half:] = (self.features[:, half:] - self.norm_stats['delta_mean'])  / self.norm_stats['delta_std']
        else:
            self.norm_stats = None

        # 预计算 per-feature std 用于 scale-aware 噪声
        self.feature_std = self.features.std(axis=0).clip(min=1e-6)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat = self.features[idx].copy()

        if self.augment:
            # 1. Scale-aware 高斯噪声：noise ~ N(0, aug_noise_std * feature_std)
            #    仅对 honest 样本施加，避免破坏攻击样本的量化特征
            if self.labels[idx] == 1:
                noise = np.random.randn(feat.shape[0]).astype(np.float32) * (self.aug_noise_std * self.feature_std)
                feat = feat + noise

            # 2. 特征 dropout：随机置零少量维度（模拟特征缺失，提高泛化）
            mask = np.random.binomial(1, 1 - self.aug_dropout_p, feat.shape[0]).astype(np.float32)
            feat = feat * mask

        return {
            'features': torch.from_numpy(feat),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'attack_type': self.attack_types[idx],
            'layer_idx': self.layer_indices[idx]
        }

    def get_layer_samples(self, layer_idx: int) -> 'DiscriminatorDataset':
        """返回指定层的子数据集"""
        layer_samples = [s for s in self.samples if s.layer_idx == layer_idx]
        return DiscriminatorDataset(layer_samples, augment=self.augment,
                                    aug_noise_std=self.aug_noise_std,
                                    aug_dropout_p=self.aug_dropout_p,
                                    normalize=self.normalize,
                                    norm_stats=self.norm_stats)

