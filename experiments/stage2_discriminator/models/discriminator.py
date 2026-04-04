"""
Discriminator Model for Stage 2
多种架构：MLP, CNN, Attention, ResNet MLP, Transformer
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


class CNNDiscriminator(nn.Module):
    """1D CNN 判别器"""

    def __init__(self, input_dim: int, channels: List[int], kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()

        self.input_dim = input_dim
        conv_layers = []
        in_channels = 1

        for out_channels in channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # 计算卷积后的维度
        conv_out_dim = input_dim // (2 ** len(channels)) * channels[-1]
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, input_dim]
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionDiscriminator(nn.Module):
    """Self-Attention 判别器"""

    def __init__(self, input_dim: int, num_heads: int = 8, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)  # [batch, 1, hidden_dim]
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(attn_out.squeeze(1))
        return self.fc(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ResNetDiscriminator(nn.Module):
    """ResNet 风格 MLP，10M-50M 级别"""

    def __init__(self, input_dim: int, hidden_dim: int = 4096, num_blocks: int = 8, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.blocks(x)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerDiscriminator(nn.Module):
    """Transformer Encoder 判别器，50M-100M 级别"""

    def __init__(self, input_dim: int, hidden_dim: int = 2048, num_heads: int = 16,
                 num_layers: int = 6, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)  # [batch, 1, hidden_dim]
        x = self.encoder(x).squeeze(1)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
