"""
Discriminator Model for Stage 2
多种架构：MLP, CNN, Attention, ResNet MLP, Transformer, DualStream, GatedDualStream, TripleStream
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def kaiming_init_(module: nn.Module):
    """对 Linear 层应用 Kaiming 初始化，对 BN 重置为标准值"""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


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


class BNResNetDiscriminator(nn.Module):
    """带 BatchNorm 的轻量 ResNet MLP，约 1-3M 参数，稳定性优于纯 LayerNorm 版本"""

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_blocks: int = 4, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.act = nn.GELU()
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.proj(x)
        for block in self.blocks:
            x = self.act(x + block(x))
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualStreamDiscriminator(nn.Module):
    """双流判别器：分别处理 x_curr 和 delta，再融合

    专为 concat([x_curr, delta]) 输入设计：
    - x_curr 流：捕捉激活值的绝对分布特征
    - delta 流：捕捉残差的量化/扰动特征
    - 融合层：联合决策
    """

    def __init__(self, half_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.half_dim = half_dim  # = hidden_dim_of_LLM (1536)

        self.curr_stream = nn.Sequential(
            nn.Linear(half_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.delta_stream = nn.Sequential(
            nn.Linear(half_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        x_curr = x[:, :self.half_dim]
        delta = x[:, self.half_dim:]
        curr_feat = self.curr_stream(x_curr)
        delta_feat = self.delta_stream(delta)
        fused = torch.cat([curr_feat, delta_feat], dim=1)
        return self.fusion(fused)

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


class GatedDualStreamDiscriminator(nn.Module):
    """门控双流判别器

    在 DualStreamDiscriminator 基础上，引入跨流门控：
    - x_curr 流学习"语义基线"
    - delta 流学习"残差异常"
    - 门控机制让 x_curr 流动态决定每个维度"要对 delta 给予多少关注"
    这样模型能自适应地在不同层、不同维度上调整两流的相对权重。
    """

    def __init__(self, half_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.half_dim = half_dim

        self.curr_enc = nn.Sequential(
            nn.Linear(half_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta_enc = nn.Sequential(
            nn.Linear(half_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 门控：由 x_curr 特征生成对 delta 特征的逐维权重
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        kaiming_init_(self)

    def forward(self, x):
        x_curr = x[:, :self.half_dim]
        delta = x[:, self.half_dim:]
        curr_feat = self.curr_enc(x_curr)
        delta_feat = self.delta_enc(delta)
        gate = self.gate(curr_feat)
        gated_delta = gate * delta_feat
        fused = torch.cat([curr_feat, gated_delta], dim=1)
        return self.fusion(fused)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TripleStreamDiscriminator(nn.Module):
    """三流判别器：x_curr + delta + delta_normalized

    针对 layer_21 的核心问题：深层激活绝对值大，量化误差
    相对于 x_curr 很小，但相对于残差 delta 仍然显著。
    引入第三流 delta_normalized = delta / (||x_prev|| + eps)，
    提供 scale-invariant 的比例性视角，与绝对 delta 形成互补。
    """

    def __init__(self, half_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.half_dim = half_dim

        def make_stream(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
            )

        self.curr_stream = make_stream(half_dim)
        self.delta_stream = make_stream(half_dim)
        self.norm_delta_stream = make_stream(half_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        kaiming_init_(self)

    def forward(self, x):
        x_curr = x[:, :self.half_dim]
        delta = x[:, self.half_dim:]
        x_prev = x_curr - delta
        x_prev_norm = x_prev.norm(dim=1, keepdim=True).clamp(min=1e-6)
        delta_normalized = delta / x_prev_norm

        f_curr = self.curr_stream(x_curr)
        f_delta = self.delta_stream(delta)
        f_norm = self.norm_delta_stream(delta_normalized)
        fused = torch.cat([f_curr, f_delta, f_norm], dim=1)
        return self.fusion(fused)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _quantization_stat_features(delta: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
    """从 delta 和 x_prev 提取对量化网格敏感的统计特征（共 12 维）

    量化噪声的核心特征：
    - 离散化导致 sorted differences 中出现周期性"跳变"（量化步长的倍数）
    - 量化使尾部被截断 → 低 kurtosis，低 max_abs/std 比
    - delta_normalized 的统计量对 scale 不敏感，专攻 layer_21

    返回 [B, 12] 的统计特征向量（均为可微操作）
    """
    eps = 1e-6

    # --- delta 的统计量（绝对视角）---
    d_std = delta.std(dim=1, keepdim=True).clamp(min=eps)
    d_mean = delta.mean(dim=1, keepdim=True)
    d_centered = delta - d_mean

    # 峰度（量化后低于正态）
    kurtosis = ((d_centered ** 4).mean(dim=1) / (d_std.squeeze(1) ** 4 + eps))  # [B]
    # 偏度
    skewness = ((d_centered ** 3).mean(dim=1) / (d_std.squeeze(1) ** 3 + eps))  # [B]
    # Max-abs / std（量化截断后该比值偏低）
    max_abs_ratio = delta.abs().max(dim=1).values / (d_std.squeeze(1) + eps)
    # 排序后相邻差分的 std（量化 → 集中在步长倍数 → 低 std）
    sorted_delta, _ = delta.abs().sort(dim=1)
    sorted_diff_std = (sorted_delta[:, 1:] - sorted_delta[:, :-1]).std(dim=1)

    # --- delta_normalized 的统计量（scale 不变，专攻 layer_21）---
    x_prev_norm = x_prev.norm(dim=1, keepdim=True).clamp(min=eps)
    dn = delta / x_prev_norm                               # [B, D]
    dn_std = dn.std(dim=1).clamp(min=eps)
    dn_mean = dn.mean(dim=1)
    dn_centered = dn - dn_mean.unsqueeze(1)

    dn_kurtosis = ((dn_centered ** 4).mean(dim=1) / (dn_std ** 4 + eps))
    dn_skewness = ((dn_centered ** 3).mean(dim=1) / (dn_std ** 3 + eps))
    dn_max_ratio = dn.abs().max(dim=1).values / (dn_std + eps)

    # --- x_prev 的模（携带 scale 信息，layer_21 的模更大）---
    x_prev_norm_scalar = x_prev_norm.squeeze(1)
    delta_norm_scalar = delta.norm(dim=1)
    # 相对扰动强度：||delta|| / ||x_prev||
    relative_perturbation = delta_norm_scalar / (x_prev_norm_scalar + eps)

    feats = torch.stack([
        kurtosis, skewness, max_abs_ratio, sorted_diff_std,
        dn_kurtosis, dn_skewness, dn_max_ratio,
        x_prev_norm_scalar.log1p(), delta_norm_scalar.log1p(),
        relative_perturbation,
        dn_std, d_std.squeeze(1),
    ], dim=1)  # [B, 12]

    # 对极端值做 clamp，防止梯度爆炸
    return feats.clamp(-100, 100)


class StatEnhancedGatedDiscriminator(nn.Module):
    """统计增强的门控双流判别器

    核心思想：在 GatedDualStream 的深度特征之外，显式计算
    12 维对量化敏感的统计特征，两路特征在融合层联合决策。
    统计流是 scale-invariant 的，专门解决 layer_21 的 scale 问题。
    """

    def __init__(self, half_dim: int, hidden_dim: int = 256, dropout: float = 0.2,
                 proj_dim: int = 128):
        super().__init__()
        self.half_dim = half_dim
        self.proj_dim = proj_dim   # 用于 SupCon 的投影头输出维度
        STAT_DIM = 12

        # 深度流（与 GatedDualStream 相同）
        self.curr_enc = nn.Sequential(
            nn.Linear(half_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.delta_enc = nn.Sequential(
            nn.Linear(half_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

        # 统计特征的小型编码器
        self.stat_enc = nn.Sequential(
            nn.Linear(STAT_DIM, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        # 融合头（分类）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

        # 投影头（用于 SupCon Loss）
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        kaiming_init_(self)

    def _encode(self, x):
        """提取融合特征向量（供分类头和投影头共用）"""
        x_curr = x[:, :self.half_dim]
        delta = x[:, self.half_dim:]
        x_prev = x_curr - delta

        curr_feat = self.curr_enc(x_curr)
        delta_feat = self.delta_enc(delta)
        gated_delta = self.gate(curr_feat) * delta_feat
        deep_feat = torch.cat([curr_feat, gated_delta], dim=1)

        stat_feat = _quantization_stat_features(delta, x_prev)
        stat_enc = self.stat_enc(stat_feat)

        return torch.cat([deep_feat, stat_enc], dim=1)

    def forward(self, x):
        return self.classifier(self._encode(x))

    def project(self, x):
        """用于 SupCon Loss 的归一化投影"""
        z = self.projector(self._encode(x))
        return F.normalize(z, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020)

    拉近同类样本的嵌入、推开不同类样本。
    对难分类样本（如 layer_21 的量化攻击）比 CE loss 提供更强的梯度信号。
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: [B, proj_dim]，已 L2 归一化
        labels:   [B]，整数类别标签
        """
        device = features.device
        B = features.shape[0]

        # 计算所有对的余弦相似度矩阵，除以温度
        sim = torch.mm(features, features.T) / self.temperature  # [B, B]

        # 屏蔽对角线（自身）
        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask_self, -1e9)

        # 同类掩码
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T) & ~mask_self  # [B, B]

        # 分母：所有非自身样本
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        # 仅对有正样本的 anchor 计算损失
        num_pos = mask_pos.float().sum(dim=1)
        valid = num_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -(log_prob * mask_pos.float()).sum(dim=1) / (num_pos + 1e-6)
        return loss[valid].mean()


def _fft_features(delta: torch.Tensor, n_feats: int = 32) -> torch.Tensor:
    """从 delta 中提取频域特征（共 n_feats*2 维）

    量化产生周期性"梳状"结构：1536 维 delta 只落在 16/64/256 个格点上，
    排序后做 FFT 会在对应频率出现尖锐谐波峰，honest 则频谱平滑衰减。

    步骤：
    1. 对 delta 的绝对值排序（去除符号噪声，只看幅度分布）
    2. 对排序后序列做 rfft，得到实数功率谱
    3. 取低频段（前 n_feats 个频率分量，包含基频和主要谐波）
    4. 归一化（除以总能量），使特征对 scale 不变

    返回 [B, n_feats*2]：归一化功率谱 + log(功率谱+1)
    """
    # 排序（让量化格点结构在频域可见）
    sorted_delta, _ = delta.abs().sort(dim=1)         # [B, D]

    # rfft 返回复数，取模得功率谱
    fft_out = torch.fft.rfft(sorted_delta, dim=1)     # [B, D//2+1]
    power = fft_out.abs()                              # [B, D//2+1]

    # 取前 n_feats 个频率分量（基频 + 谐波区域）
    power = power[:, :n_feats]                         # [B, n_feats]

    # Scale-invariant 归一化：除以总能量
    total_energy = power.sum(dim=1, keepdim=True).clamp(min=1e-6)
    power_norm = power / total_energy                  # [B, n_feats]

    # log 变换增强小峰的可见性
    power_log = torch.log1p(power_norm * 100)          # [B, n_feats]

    return torch.cat([power_norm, power_log], dim=1)   # [B, n_feats*2]


class FFTEnhancedDiscriminator(nn.Module):
    """频域增强判别器

    在 GatedDualStream 基础上加入 FFT 频域特征流：
    - 深度流（GatedDualStream）：学习激活的语义特征
    - 统计流（StatEnhanced）：捕捉量化的统计指纹
    - 频域流（FFT）：捕捉量化的周期性谐波结构（scale-invariant）

    三路融合，对 layer_21 的 scale 问题有针对性改善。
    """

    def __init__(self, half_dim: int, hidden_dim: int = 256, dropout: float = 0.2,
                 proj_dim: int = 128, n_fft_feats: int = 32):
        super().__init__()
        self.half_dim = half_dim
        self.proj_dim = proj_dim
        self.n_fft_feats = n_fft_feats
        STAT_DIM = 12
        FFT_DIM = n_fft_feats * 2

        # 深度流（GatedDualStream）
        self.curr_enc = nn.Sequential(
            nn.Linear(half_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.delta_enc = nn.Sequential(
            nn.Linear(half_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

        # 统计特征流（12维）
        self.stat_enc = nn.Sequential(
            nn.Linear(STAT_DIM, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
        )

        # 频域特征流（n_fft_feats*2 维）
        self.fft_enc = nn.Sequential(
            nn.Linear(FFT_DIM, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
        )

        fusion_dim = hidden_dim * 2 + 64 + 64   # deep + stat + fft
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.projector = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )
        kaiming_init_(self)

    def _encode(self, x):
        x_curr = x[:, :self.half_dim]
        delta = x[:, self.half_dim:]
        x_prev = x_curr - delta

        # 深度流
        curr_feat = self.curr_enc(x_curr)
        delta_feat = self.delta_enc(delta)
        gated_delta = self.gate(curr_feat) * delta_feat
        deep_feat = torch.cat([curr_feat, gated_delta], dim=1)

        # 统计特征流
        stat_feat = _quantization_stat_features(delta, x_prev)
        stat_enc = self.stat_enc(stat_feat)

        # 频域特征流
        fft_feat = _fft_features(delta, self.n_fft_feats)
        fft_enc = self.fft_enc(fft_feat)

        return torch.cat([deep_feat, stat_enc, fft_enc], dim=1)

    def forward(self, x):
        return self.classifier(self._encode(x))

    def project(self, x):
        return F.normalize(self.projector(self._encode(x)), dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


