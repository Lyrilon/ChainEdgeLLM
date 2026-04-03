"""
PyTorch Dataset for Stage 2 Discriminator Training
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../stage1_separability'))

from data_generator import Sample


class DiscriminatorDataset(Dataset):
    """判别器训练数据集"""

    def __init__(self, samples: List[Sample]):
        self.samples = samples

        # 提取特征和标签
        self.features = []
        self.labels = []

        for sample in samples:
            self.features.append(sample.x_curr)
            # 二分类：honest=1, attack=0
            label = 1 if sample.label == 'honest' else 0
            self.labels.append(label)

        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            'features': torch.from_numpy(self.features[idx]),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
