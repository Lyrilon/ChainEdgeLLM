# Stage 2: Neural Discriminator Experiment

## 概述

Stage 2 实现了神经判别器 $D_\theta^{(k)}$ 来检测 Stage 1 无法识别的复杂攻击：
- Identity Forgery (Layer Skipping)
- Silent Precision Downgrade
- Adversarial Perturbation

## 目录结构

```
stage2_discriminator/
├── config.yaml              # 配置文件
├── run_experiment.py        # 主实验流程
├── data/                    # 数据模块
│   ├── attack_generator.py  # 攻击生成器
│   └── dataset.py           # PyTorch Dataset
├── models/                  # 模型模块
│   └── discriminator.py     # 判别器
├── training/                # 训练模块
│   ├── trainer.py           # 训练器
│   └── evaluator.py         # 评估器
└── results/                 # 输出目录
```

## 使用方法

```bash
cd experiments/stage2_discriminator
python run_experiment.py
```

## 判别器架构

- **Large**: 1536→1024→512→256→128→2 (2.5M params)
- **Medium**: 1536→512→256→128→2 (0.8M params)
- **Small**: 1536→256→128→64→2 (0.5M params)

## 攻击类型

1. **Layer Skipping**: $\tilde{x}_k = x_{k-1}$
2. **Precision Downgrade**: FP16/INT8/INT4 量化
3. **Adversarial Perturbation**: FGSM 扰动

## 输出

- `results/{arch_name}/best_model.pt`: 最佳模型
- `results/results.json`: 所有架构的性能对比
