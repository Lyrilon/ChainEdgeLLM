# Stage 2 判别器探索日志

**日期**: 2026-04-07  
**目标**: 训练 TS-ZRV Stage 2 的神经判别器 $D_\theta^{(k)}$，检测 layer_skipping、precision_downgrade、adversarial_perturbation 三类攻击  
**模型**: qwen2.5-1.5b-instruct（hidden_dim=1536，目标层 7/14/21）

---

## 问题背景

Stage 2 判别器接收中间层激活 $x_k$，输出 honest/dishonest 二分类。初始实验（只用 `x_curr` 作为输入特征）在 `precision_downgrade` 攻击上的错误率高达 **50~60%**，相当于随机猜。

---

## 根本原因分析

### 问题 1：判别器只看 `x_curr`，缺失参照信号

```python
# 错误做法
self.features.append(sample.x_curr)
```

`precision_downgrade` 的本质是对 `x_curr` 施加轻微扰动。**量化误差在 x_curr 空间只有 3%，但在 delta 空间高达 200%**：

| 空间 | 信号范围 | INT4 量化误差 | 误差/信号比 |
|------|---------|-------------|------------|
| x_curr | [-15, +15]，范围 30 | ±1.0 | **3%**（被淹没） |
| delta = x_curr - x_prev | [-0.5, +0.5]，范围 1 | ±1.0 | **200%**（清晰可见） |

### 问题 2：fp16 伪攻击污染训练数据

`bit_widths=[16, 8, 4]` 中 fp16 的误差量级约 `1e-4`，与 honest 几乎相同，导致判别器收到矛盾的训练信号（同分布样本标签不同），结果 ~50% 错误率相当于随机猜。

---

## 改进历程

### v1 — 差分特征 + 去除 fp16（`dataset.py`、`config.yaml`）

**核心改动**：
```python
# 修复后
delta = sample.x_curr - sample.x_prev
feature = np.concatenate([sample.x_curr, delta])  # input_dim: 1536 → 3072
```
- 去除 `bit_widths` 中的 16（fp16）
- `run_experiment.py` 所有架构的 `input_dim` 改为 `hidden_dim * 2`

**结果**：precision_downgrade 从 55% 降至 ~37%

---

### v2 — batch_size 优化（`config.yaml`）

GPU benchmark 测试结果：
```
batch_size=64   →  22,182 samples/s
batch_size=4096 → 1,530,460 samples/s  (69× 提升，显存仅 123MB)
```
- `batch_size: 64 → 4096`
- LR 按 sqrt scaling: `1e-4 × sqrt(4096/64) = 8e-4`
- 新增 warmup 5 epochs + cosine annealing + grad_clip=1.0

**结果**：训练速度大幅提升，但 LR 过高导致 layer_21 部分模型不稳定

---

### v3 — DualStreamDiscriminator（`models/discriminator.py`）

**核心架构**：
```python
class DualStreamDiscriminator(nn.Module):
    """x_curr 和 delta 走独立 stream，再融合"""
    # x_curr stream: 捕捉语义基线
    # delta stream: 捕捉残差异常
    # fusion: 联合决策
```

**结果**（layer_14 最佳）：
- precision_downgrade: 55% → **6%**
- adversarial_perturbation: 45% → **0.3%**

---

### v4 — GatedDualStream + TripleStream（`models/discriminator.py`）

**GatedDualStreamDiscriminator**：x_curr 流生成门控权重，动态控制对 delta 流各维度的关注程度：
```python
gate = self.gate(curr_feat)       # [B, hidden_dim]，sigmoid
gated_delta = gate * delta_feat   # 自适应加权
```

**TripleStreamDiscriminator**：针对 layer_21 的 scale 问题，增加第三流：
```python
x_prev_norm = x_prev.norm(dim=1, keepdim=True).clamp(min=1e-6)
delta_normalized = delta / x_prev_norm   # scale-invariant
```

**layer_21 问题根源**：深层激活绝对值大（layer_21 norm >> layer_7 norm），量化误差相对 x_curr 很小但相对 delta 仍显著。`delta_normalized` 提供了比例性视角。

**结果**：
- layer_7 precision_downgrade: **16.9%**（gated_dual_m）
- layer_14 precision_downgrade: **3.0%**（gated_dual_m）
- layer_21 precision_downgrade: **41.5%**（triple_stream_m）

---

### v5 — StatEnhancedGatedDiscriminator + SupConLoss

**StatEnhancedGatedDiscriminator**：在门控双流基础上，显式计算 12 维对量化敏感的统计特征：

```python
def _quantization_stat_features(delta, x_prev):
    """12维统计特征（均为可微操作）"""
    # 量化指纹：kurtosis、skewness、max_abs/std、sorted_diff_std
    # scale-invariant 视角：delta_normalized 的统计量
    # 信号强度：||x_prev||、||delta||、relative_perturbation
```

**为什么统计特征有效**：INT4 量化产生离散化的"量化网格"，使得：
- kurtosis 降低（尾部被截断）
- sorted_diff_std 降低（值集中在步长倍数上）
- delta_normalized 统计量对 scale 不变

**SupConLoss**（Supervised Contrastive Learning，Khosla et al. 2020）：
```python
class SupConLoss(nn.Module):
    """拉近同类嵌入、推开不同类嵌入
    对难分样本比 CE loss 提供更强的梯度信号
    temperature=0.07，layer_21 专用
    """
```

**per_layer_training** 配置：layer_21 使用 `supcon_ce` 联合损失（60% CE + 40% SupCon）

**结果**：layer_21 precision_downgrade: 41.5% → **33.4%**

---

### v6 — 三路并进：数据量 × 2 + 数据增强 + INT6 + 集成

**1. 扩大数据**：`num_samples: 2000 → 4000`

**2. 数据增强**（仅 layer_21 训练集）：
```python
class DiscriminatorDataset(Dataset):
    # augment=True 时：
    # - honest 样本加 scale-aware 高斯噪声
    # - 所有样本 3% 特征 dropout
```
增强只施加于 honest 样本，避免破坏攻击样本的量化特征。

**3. INT6 量化**：`bit_widths: [8, 4] → [6, 8, 4]`，覆盖中等量化强度（误差介于 INT8 和 INT4 之间）

**4. 集成评估**（EnsembleEvaluator）：每层训练完后对所有架构做 soft voting

**顺带修复**：发现并修复了 `model_loader.py` 中 `input_ids` 未自动转到 GPU 的 bug（数据量从 2000 扩大触发缓存未命中时才暴露）：
```python
device = next(self.model.parameters()).device
input_ids = input_ids.to(device)
```

**结果**（v6 最佳）：
- layer_7 precision_downgrade: **5.11%**（stat_enhanced_m），集成 5.74%
- layer_14 precision_downgrade: **1.15%**（stat_enhanced_s），集成 **1.44%**
- layer_21 precision_downgrade: **18.4%**（triple_stream_m）

---

### v7 — 继续扩大数据量 8000 样本

**结果**：

| 架构 | layer_7 | layer_14 | layer_21 |
|------|:-------:|:--------:|:--------:|
| gated_dual_m | 6.97% | 0.47% | 27.73% |
| triple_stream_m | 5.90% | 0.77% | **18.59%** ✨ |
| stat_enhanced_s | 5.38% | 0.69% | 19.74% |
| stat_enhanced_m | 5.38% | 0.55% | 19.96% |
| **集成（4模型）** | 5.74% | **0.33%** 🎉 | 20.68% |

adversarial_perturbation 结果：

| 架构 | layer_7 | layer_14 | layer_21 |
|------|:-------:|:--------:|:--------:|
| stat_enhanced_m | **0.48%** | **0.03%** | 31.09% |
| 集成 | 0.22% | 0.03% | 32.01% |

---

### v8 — 8000 样本全架构对比（7 架构穷举）

v7 只跑了 4 个架构，v8 补全所有历史最优架构在 8000 样本下的完整对比。

#### precision_downgrade 错误率

| 架构 | layer_7 | layer_14 | layer_21 |
|------|:-------:|:--------:|:--------:|
| dual_stream_m | 6.32% | 0.85% | 22.38% |
| gated_dual_s | 6.51% | 0.66% | **16.53%** ✨ |
| gated_dual_m | 7.11% | 0.66% | 21.55% |
| triple_stream_s | 9.14% | 1.15% | 19.80% |
| triple_stream_m | 8.26% | 0.82% | 18.92% |
| stat_enhanced_s | 8.68% | 1.15% | 21.09% |
| **stat_enhanced_m** | **5.85%** ✨ | **0.49%** ✨ | 20.32% |
| **集成（7模型）** | 7.61% | 0.60% | 20.26% |

#### adversarial_perturbation 错误率

| 架构 | layer_7 | layer_14 | layer_21 |
|------|:-------:|:--------:|:--------:|
| triple_stream_m | 0.65% | 0.11% | 30.97% |
| **stat_enhanced_m** | **0.67%** | **0.03%** | 31.73% |
| gated_dual_m | 0.76% | 0.03% | 32.01% |
| **集成** | 0.62% | 0.03% | 32.21% |

**关键发现**：`gated_dual_s`（小型门控双流）在 layer_21 上以 **16.53%** 胜出，比 `gated_dual_m` 还好，说明 layer_21 上**小模型更不容易过拟合**，更大的模型反而在数据量有限时容易记住训练集分布而泛化不足。

---

## 全历程对比（precision_downgrade 最佳值）

| 版本 | 数据量 | layer_7 | layer_14 | layer_21 |
|------|:------:|:-------:|:--------:|:--------:|
| 初始基线 | 2000 | 55% | 52% | 60% |
| v1 差分特征 | 2000 | 36% | 31% | 51% |
| v2 batch优化 | 2000 | 37% | 30% | 48% |
| v3 DualStream | 2000 | 22% | 6% | 43% |
| v4 GatedDual+Triple | 2000 | 16.9% | 3.0% | 41.5% |
| v5 StatEnhanced+SupCon | 2000 | 18.3% | 4.8% | 33.4% |
| v6 4000样本+增强+INT6 | 4000 | 5.11% | 1.15% | 18.4% |
| v7 8000样本（4架构） | 8000 | 5.38% | 0.47% | 18.59% |
| **v8 8000样本（7架构全比）** | **8000** | **5.85%** | **0.49%** | **16.53%** 🎉 |
| **v8 集成（7模型）** | 8000 | 7.61% | **0.60%** | 20.26% |

---

## 关键结论

### 1. 数据量是最大杠杆
从 2000 → 4000 → 8000 样本，layer_21 precision_downgrade 分别为 33% → 18% → 18.6%。**4000→8000 的边际收益明显缩减**，说明 layer_21 存在真正的信息上限，单纯扩数据已进入收益递减区。

### 2. 差分特征是根本性突破
`concat([x_curr, delta])` 是本次探索中单一改动效果最显著的：量化噪声在 delta 空间的信噪比是 x_curr 空间的 67 倍。

### 3. 架构演进对 layer_7/14 效果显著，对 layer_21 有限
DualStream → GatedDualStream → TripleStream → StatEnhanced 这条架构演进路线对 layer_7/14 将错误率从 ~30% 降到 ~1%，但对 layer_21 从 ~40% 只降到 ~18%。**layer_21 更多依赖数据量，而非架构复杂度**。

### 4. layer_21 的天花板
AUC 稳定在 0.91~0.93，precision_downgrade 在 18% 左右停滞。推测原因：
- 深层激活是多层非线性变换的叠加，量化痕迹被"消化"得更彻底
- adversarial_perturbation 的错误率（~30%）也明显高于其他层
- 这可能是 layer_21 信号本质上更难与 honest 区分的边界

### 5. 集成对 layer_14 非常有效
layer_14 集成（4模型 soft voting）precision_downgrade 达到 **0.33%**，adversarial_perturbation **0.03%**，接近完美检测。

---

## 最终推荐配置

**最终推荐配置**（每层选最佳单模型）：

| 层 | 推荐架构 | precision_downgrade | adversarial_perturbation |
|----|---------|:-------------------:|:------------------------:|
| layer_7 | stat_enhanced_m | **5.85%** | 0.67% |
| layer_14 | stat_enhanced_m | **0.49%** | 0.03% |
| layer_21 | gated_dual_s | **16.53%** | 28.73% |

---

## 代码结构变更记录

```
experiments/stage2_discriminator/
├── models/discriminator.py     # 新增: DualStream, GatedDualStream, TripleStream,
│                               #       BNResNet, StatEnhancedGated, SupConLoss
├── data/dataset.py             # 新增: augment参数（高斯噪声+特征dropout）
├── data/attack_generator.py    # 新增: INT6量化（bits=6）
├── training/trainer.py         # 新增: warmup+cosine LR, grad_clip, 
│                               #       label_smoothing, supcon_ce联合损失, tqdm进度条
├── training/ensemble_evaluator.py  # 新增: 多模型soft voting集成评估
└── run_experiment.py           # 新增: per_layer_training配置, 
│                               #       增强数据集, 训练后集成评估

experiments/stage1_separability/
└── model_loader.py             # 修复: input_ids.to(device) bug
```

---

## 未来方向

1. **layer_21 专项**：尝试在 layer_21 上做更激进的 mixup 数据增强，或引入 temperature scaling 改善校准
2. **更多量化类型**：模拟 GPTQ、AWQ 等实际量化算法的激活分布，使训练攻击样本更贴近真实攻击
3. **知识蒸馏**：用 layer_14 的高精度判别器指导 layer_21 判别器的训练（教师-学生框架）
4. **在线更新**：研究 $D_\theta^{(k)}$ 在部署后接收新激活样本时的在线微调机制
