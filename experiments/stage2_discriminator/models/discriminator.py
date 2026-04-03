"""
Discriminator Model for Stage 2
可配置的 MLP 判别器
"""

import torch
import torch.nn as nn
from typing import List


class Discriminator(nn.Module):
    """MLP 判别器"""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
