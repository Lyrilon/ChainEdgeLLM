# Stage 1 Separability Experiment for TS-ZRV

TS-ZRV (Two-Stage Zero-Recomputation Verification) 阶段一可分性实验

## 简介

本实验用于验证论文中提出的 **Stage 1: Causal Binding Check** 的可分性。通过余弦相似度检测两种威胁向量:

1. **Random Noise (Byzantine Failure)** - 随机噪声攻击
2. **Replay Attack (Stealing Output)** - 重放攻击

核心假设: 在 Transformer 的残差流中，合法前向传播保持输入和输出之间的因果关联，而攻击样本破坏这种关联。

## 快速开始

### 1. 安装依赖

```bash
# 使用 pip 安装依赖
pip install -r requirements.txt

# 或使用 conda
conda install pytorch transformers numpy scipy matplotlib seaborn pyyaml tqdm
pip install modelscope
```

### 2. 快速测试

```bash
# 快速测试模式 (小规模数据，无需 GPU)
python run_experiment.py --quick-test
```

### 3. 完整实验

```bash
# 编辑配置文件 config.yaml
# 修改模型、数据集、攻击类型等参数

# 运行完整实验
python run_experiment.py --config config.yaml
```

## 项目结构

```
stage1_separability/
├── config.yaml              # 实验配置文件
├── requirements.txt         # Python 依赖
├── run_experiment.py        # 主实验脚本
├── model_loader.py          # 模型加载器 (ModelScope)
├── data_generator.py        # 数据生成器
├── similarity_analyzer.py   # 相似度分析器
├── visualizer.py            # 可视化器
├── tests/                   # 单元测试
│   └── test_similarity.py
└── README.md               # 本文件
```

## 实验流程

1. **加载模型**: 从 ModelScope (魔塔) 下载并加载预训练模型
2. **加载数据**: 从 ModelScope 加载数据集 (WikiText, C4 等)
3. **生成样本**:
   - 诚实样本: 正常前向传播生成 (x_{k-1}, x_k) 对
   - 攻击样本: 随机噪声、重放攻击等
4. **计算相似度**: 计算余弦相似度 S(x_{k-1}, x_k)
5. **分析可分性**: 统计分析、ROC-AUC、分离间隙等
6. **生成可视化**: 分布图、ROC曲线、阈值扫描等

## 模型和数据集 (ModelScope 魔塔)

### 支持的模型

| 模型 | ModelScope ID |
|------|--------------|
| GPT-2 | AI-ModelScope/gpt2 |
| GPT-2 Medium | AI-ModelScope/gpt2-medium |
| LLaMA-2-7B | modelscope/Llama-2-7b-ms |
| Qwen2.5-1.5B | qwen/Qwen2.5-1.5B-Instruct |

### 支持的数据集

| 数据集 | ModelScope ID |
|--------|--------------|
| WikiText-2 | AI-ModelScope/wikitext-2-raw-v1 |
| WikiText-103 | AI-ModelScope/wikitext-103-raw-v1 |
| C4 | AI-ModelScope/c4 |

## 配置说明

编辑 `config.yaml`:

```yaml
# 模型配置
model:
  name: "gpt2"  # 模型名称
  target_layers: [4, 8, 12]  # 测试的层

# 数据集配置
dataset:
  name: "wikitext"
  num_samples: 2000  # 样本数

# 攻击类型配置
attacks:
  random_noise:
    enabled: true
    noise_levels: [0.1, 0.5, 1.0, 2.0, 5.0]  # 噪声标准差

  replay_attack:
    enabled: true
    replay_types:
      - "cross_sequence_same_layer"
      - "cross_sequence_cross_layer"
```

## 实验结果

运行完成后，结果保存在 `results/` 目录:

- `analysis_report.json` - 完整分析报告
- `similarity_results.json` - 原始相似度数据
- `distribution_histogram*.png` - 分布直方图
- `box_violin*.png` - 箱线图和小提琴图
- `roc_curves.png` - ROC 曲线
- `pr_curves.png` - PR 曲线
- `threshold_sweep.png` - 阈值扫描图
- `layer_comparison.png` - 层间比较

## 关键指标

### 1. 分离间隙 (Separation Gap)

```
Δ = min(S_honest) - max(S_malicious)
```

- Δ > 0: 可分性良好
- Δ ≤ 0: 存在重叠，需要调整

### 2. ROC-AUC

- AUC > 0.95: 优秀
- 0.8 < AUC ≤ 0.95: 良好
- AUC ≤ 0.8: 需要改进

### 3. 推荐阈值

通过 F1 分数优化得到的最佳阈值 τ_k

## 单元测试

```bash
# 运行所有测试
python -m pytest tests/

# 或直接使用 unittest
python tests/test_similarity.py
```

## 引用

如果本实验对您的研究有帮助，请引用:

```bibtex
@article{chainedgellm2025,
  title={Blockchain-based Trustworthy Collaborative LLM Inference over Edge Devices},
  author={ChainEdgeLLM Team},
  journal={IEEE Conference},
  year={2025}
}
```

## 许可证

MIT License
