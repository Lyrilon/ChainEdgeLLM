# Stage 2 判别器 SOTA 配置

**项目**: ChainEdgeLLM — TS-ZRV Stage 2 Neural Discriminator  
**基础模型**: qwen2.5-1.5b-instruct（hidden_dim=1536）  
**任务**: 二分类（honest vs. attack），检测 precision_downgrade / adversarial_perturbation / layer_skipping 三类攻击  
**最终数据量**: 8000 条 honest 样本（alpaca 数据集），攻击样本通过规则生成（INT4/INT6/INT8 量化、FGSM 扰动、层跳过）

---

## Layer 7 SOTA

### 架构：StatEnhancedGatedDiscriminator（stat_enhanced_m）

**实验版本**: v8（8000 samples, 7-arch full comparison）  
**参数量**: 3,223,042

#### 架构设计

三路融合架构，输入为 `concat([x_curr, delta])` （dim=3072）：

```
输入 [B, 3072]
    │
    ├─── 深度流（GatedDualStream）──────────────────────────────┐
    │     x_curr → Linear(1536,512) → LN → GELU → Dropout      │
    │     delta  → Linear(1536,512) → LN → GELU → Dropout      │
    │     gate   = Sigmoid(Linear(512,512))(curr_feat)           │
    │     gated_delta = gate × delta_feat                        │
    │     output: concat([curr_feat, gated_delta]) [B, 1024]    │
    │                                                            │
    ├─── 统计特征流（12维）──────────────────────────────────────┤
    │     kurtosis, skewness, max_abs/std, sorted_diff_std       │
    │     + delta_normalized 的对应统计量（scale-invariant）      │
    │     + ||x_prev||, ||delta||, relative_perturbation         │
    │     → Linear(12,64) → LN → GELU → Linear(64,64) [B, 64]  │
    │                                                            │
    └─── 融合分类头 ─────────────────────────────────────────────┘
          concat([deep(1024), stat(64)]) → Linear(1088,512)
          → LN → GELU → Dropout → Linear(512,256) → GELU
          → Linear(256,2)
```

#### 训练配置

| 参数 | 值 |
|------|---|
| 数据量 | 8000 honest samples |
| Batch size | 4096 |
| Learning rate | 8e-4（sqrt scaling） |
| LR schedule | Warmup 5 epochs + Cosine annealing |
| Grad clip | 1.0 |
| Weight decay | 1e-4 |
| Class weights | [1, 7]（攻击:honest） |
| Label smoothing | 0.05 |
| Early stopping patience | 15 epochs |
| Max epochs | 80 |
| Loss | CrossEntropy |
| Data augmentation | 无（layer_7 不启用） |

#### 测试集性能（layer 7）

| 攻击类型 | 样本数 | 错误数 | **错误率** |
|---------|:-----:|:-----:|:---------:|
| precision_downgrade | 3,642 | 213 | **5.85%** |
| adversarial_perturbation | 3,561 | 24 | **0.67%** |
| layer_skipping | 1,179 | 0 | **0.00%** |
| honest（误报） | 1,218 | 42 | **3.45%** |

| 综合指标 | 值 |
|---------|---|
| Accuracy | 97.09% |
| F1 | 0.8940 |
| AUC | **0.9918** |

---

## Layer 14 SOTA

### 架构：StatEnhancedGatedDiscriminator（stat_enhanced_m）

**实验版本**: v8（8000 samples, 7-arch full comparison）  
**参数量**: 3,223,042（同 layer_7，独立训练权重）

#### 架构设计

与 Layer 7 SOTA 相同（见上），独立训练。

#### 训练配置

与 Layer 7 完全相同（无 per-layer 覆盖）。

#### 测试集性能（layer 14）

| 攻击类型 | 样本数 | 错误数 | **错误率** |
|---------|:-----:|:-----:|:---------:|
| precision_downgrade | 3,642 | 18 | **0.49%** |
| adversarial_perturbation | 3,561 | 1 | **0.03%** |
| layer_skipping | 1,179 | 0 | **0.00%** |
| honest（误报） | 1,218 | 2 | **0.16%** |

| 综合指标 | 值 |
|---------|---|
| Accuracy | 99.78% |
| F1 | 0.9914 |
| AUC | **1.0000** |

> Layer 14 是三层中信号最清晰的层，接近完美检测。

---

## Layer 21 SOTA

### 架构：FFTEnhancedDiscriminator（fft_enhanced_s）

**实验版本**: v9（方向二：频域特征，8000 samples）  
**参数量**: 1,262,082

#### 架构设计

在 GatedDualStream 基础上新增频域特征流，三路融合：

```
输入 [B, 3072]
    │
    ├─── 深度流（GatedDualStream）──────────────────────────────┐
    │     （同上，hidden_dim=256）                               │
    │     output: concat([curr_feat, gated_delta]) [B, 512]    │
    │                                                            │
    ├─── 统计特征流（12维）──────────────────────────────────────┤
    │     → Linear(12,64) → LN → GELU → Linear(64,64) [B, 64] │
    │                                                            │
    ├─── 频域特征流（64维）──────────────────────────────────────┤
    │     delta.abs().sort() → rfft → 取前32个频率分量           │
    │     归一化（÷总能量）+ log1p变换 → concat [B, 64]          │
    │     → Linear(64,64) → LN → GELU → Linear(64,64) [B, 64] │
    │                                                            │
    └─── 融合分类头 ─────────────────────────────────────────────┘
          concat([deep(512), stat(64), fft(64)]) → Linear(640,256)
          → LN → GELU → Dropout → Linear(256,128) → GELU
          → Linear(128,2)
```

**频域特征的设计原理**：INT4/INT6/INT8 量化使 1536 维 delta 只落在有限的离散格点（16/64/256 个）上，排序后做 FFT 会在固定频率出现尖锐的谐波峰；honest 激活的频谱则平滑衰减。此特征对激活绝对值的 scale **完全不敏感**，专门解决 layer_21 的 scale 问题。

#### 训练配置

| 参数 | 值 |
|------|---|
| 数据量 | 8000 honest samples |
| Batch size | 4096 |
| Learning rate | 5e-4（per-layer override） |
| LR schedule | Warmup 5 epochs + Cosine annealing |
| Grad clip | 1.0 |
| Weight decay | 1e-4 |
| Class weights | [1, 5]（per-layer override） |
| Label smoothing | 0.05 |
| Early stopping patience | 25 epochs（per-layer override） |
| Max epochs | 120（per-layer override） |
| Loss | supcon_ce（60% CE + 40% SupCon，T=0.07） |
| Data augmentation | ✅ 启用（scale-aware noise + 3% feature dropout，仅 honest 样本） |
| Feature normalization | ❌ 不启用（z-score 会破坏量化步长的绝对幅度信息） |

#### 测试集性能（layer 21）

| 攻击类型 | 样本数 | 错误数 | **错误率** |
|---------|:-----:|:-----:|:---------:|
| precision_downgrade | 3,642 | 431 | **11.83%** |
| adversarial_perturbation | 3,561 | 812 | **22.80%** |
| layer_skipping | 1,179 | 0 | **0.00%** |
| honest（误报） | 1,218 | — | — |

| 综合指标 | 值 |
|---------|---|
| Accuracy | 85.57% |
| F1 | 0.6084 |
| AUC | **0.9432** |

---

## 三层汇总

| 层 | 最佳架构 | 参数量 | precision_downgrade | adversarial_perturbation | layer_skipping | AUC |
|----|---------|:-----:|:-------------------:|:------------------------:|:--------------:|:---:|
| **layer_7** | stat_enhanced_m | 3.2M | **5.85%** | **0.67%** | **0.00%** | 0.9918 |
| **layer_14** | stat_enhanced_m | 3.2M | **0.49%** | **0.03%** | **0.00%** | 1.0000 |
| **layer_21** | fft_enhanced_s | 1.3M | **11.83%** | **22.80%** | **0.00%** | 0.9432 |

---

## 关键设计决策说明

### 为何使用 `concat([x_curr, delta])` 而非单独 `x_curr`

量化误差在 delta 空间的信噪比是 x_curr 空间的约 67 倍：
- x_curr 范围 ~30，INT4 量化误差 ~±1.0 → 误差/信号 = **3%**
- delta 范围 ~1，INT4 量化误差 ~±1.0 → 误差/信号 = **200%**

### 为何 layer_21 更难

layer_21 的激活经过更多层非线性变换，量化痕迹被"消化"得更彻底。实验数据显示 AUC 稳定在 0.92~0.94，而 layer_7/14 可达 0.99~1.00，说明存在真正的信息边界。

### 为何 layer_21 用小模型（fft_enhanced_s vs m）

layer_21 数据难度最高，小模型（1.3M）比大模型（~3M）有更好的泛化——大模型容易过拟合训练集的噪声分布。这在本项目的多轮实验中一致验证（gated_dual_s 优于 gated_dual_m，fft_enhanced_s 优于 fft_enhanced_m）。

### 为何不用特征归一化（方向三失败原因）

z-score 归一化虽然消除了 x_curr 的 scale 差异，但同时压缩了 delta 中量化步长的绝对幅度信息——INT4 的量化步长 ~±1.0 是一个绝对值，归一化后变成了相对数字，失去区分度，导致性能下降 4~8pp。

---

## 模型文件位置

```
experiments/stage2_discriminator/results/
├── 20260407_203327/          # v8：layer_7 和 layer_14 SOTA
│   ├── layer_7/stat_enhanced_m/best_model.pt
│   └── layer_14/stat_enhanced_m/best_model.pt
└── 20260407_213405/          # v9：layer_21 SOTA (fft_enhanced_s)
    └── layer_21/fft_enhanced_s/best_model.pt
```
