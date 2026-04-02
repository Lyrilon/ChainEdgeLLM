"""
Stage 1 Separability Experiment for TS-ZRV
TS-ZRV 阶段一可分性实验模块

本模块用于测试 TS-ZRV (Two-Stage Zero-Recomputation Verification) 中
Stage 1 (Causal Binding Check) 的可分性。

主要功能:
- 从 ModelScope (魔塔) 加载模型和数据集
- 生成诚实样本和各种攻击样本
- 计算余弦相似度
- 分析可分性并生成可视化

作者: ChainEdgeLLM Team
"""

__version__ = "1.0.0"
__author__ = "ChainEdgeLLM Team"

from .model_loader import ModelLoader, HiddenStateExtractor
from .data_generator import (
    Sample,
    HonestSampleGenerator,
    AttackSampleGenerator,
    DatasetLoader
)
from .similarity_analyzer import (
    CosineSimilarityCalculator,
    SeparabilityAnalyzer,
    SimilarityResult
)
from .visualizer import ResultVisualizer

__all__ = [
    'ModelLoader',
    'HiddenStateExtractor',
    'Sample',
    'HonestSampleGenerator',
    'AttackSampleGenerator',
    'DatasetLoader',
    'CosineSimilarityCalculator',
    'SeparabilityAnalyzer',
    'SimilarityResult',
    'ResultVisualizer',
]
