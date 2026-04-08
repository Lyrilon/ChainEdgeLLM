"""
Stage 2 Attack Generators
生成 Identity Forgery, Precision Downgrade, Adversarial Perturbation 攻击
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../stage1_separability'))

from data_generator import Sample
from typing import List
import logging

logger = logging.getLogger(__name__)


class LayerSkippingGenerator:
    """层跳过攻击生成器（Identity Forgery）"""

    def __init__(self, honest_samples: List[Sample]):
        self.honest_samples = honest_samples

    def generate(self) -> List[Sample]:
        """生成层跳过攻击：直接返回输入"""
        attack_samples = []

        for sample in self.honest_samples:
            attack_sample = Sample(
                sample_id=sample.sample_id,
                layer_idx=sample.layer_idx,
                x_prev=sample.x_prev.copy(),
                x_curr=sample.x_prev.copy(),  # 关键：x_curr = x_prev
                label='layer_skipping',
                metadata={
                    **sample.metadata,
                    'attack_type': 'layer_skipping'
                }
            )
            attack_samples.append(attack_sample)

        logger.info(f"生成了 {len(attack_samples)} 个层跳过攻击样本")
        return attack_samples


class PrecisionDowngradeGenerator:
    """精度降级攻击生成器"""

    def __init__(self, honest_samples: List[Sample], bit_widths: List[int] = [16, 8, 4]):
        self.honest_samples = honest_samples
        self.bit_widths = bit_widths

    def quantize_dequantize(self, x: np.ndarray, bits: int) -> np.ndarray:
        """量化后反量化"""
        if bits == 16:
            return x.astype(np.float16).astype(np.float32)
        elif bits == 6:
            x_min, x_max = x.min(), x.max()
            scale = (x_max - x_min) / 63.0
            quantized = np.round((x - x_min) / scale).clip(0, 63).astype(np.uint8)
            return quantized.astype(np.float32) * scale + x_min
        elif bits == 8:
            x_min, x_max = x.min(), x.max()
            scale = (x_max - x_min) / 255.0
            quantized = np.round((x - x_min) / scale).astype(np.uint8)
            return quantized.astype(np.float32) * scale + x_min
        elif bits == 4:
            x_min, x_max = x.min(), x.max()
            scale = (x_max - x_min) / 15.0
            quantized = np.round((x - x_min) / scale).clip(0, 15).astype(np.uint8)
            return quantized.astype(np.float32) * scale + x_min
        return x

    def generate(self) -> List[Sample]:
        """生成精度降级攻击"""
        attack_samples = []

        for sample in self.honest_samples:
            for bits in self.bit_widths:
                x_degraded = self.quantize_dequantize(sample.x_curr.copy(), bits)

                attack_sample = Sample(
                    sample_id=sample.sample_id,
                    layer_idx=sample.layer_idx,
                    x_prev=sample.x_prev.copy(),
                    x_curr=x_degraded,
                    label=f'precision_downgrade_{bits}bit',
                    metadata={
                        **sample.metadata,
                        'attack_type': 'precision_downgrade',
                        'bit_width': bits
                    }
                )
                attack_samples.append(attack_sample)

        logger.info(f"生成了 {len(attack_samples)} 个精度降级攻击样本")
        return attack_samples


class AdversarialPerturbationGenerator:
    """对抗扰动攻击生成器"""

    def __init__(self, honest_samples: List[Sample], epsilon: List[float] = [0.01, 0.05, 0.1]):
        self.honest_samples = honest_samples
        self.epsilon = epsilon

    def fgsm_perturbation(self, x: np.ndarray, epsilon: float) -> np.ndarray:
        """FGSM 扰动：x' = x + ε·sign(random_gradient)"""
        gradient = np.random.randn(*x.shape)
        perturbation = epsilon * np.sign(gradient)
        return x + perturbation

    def generate(self) -> List[Sample]:
        """生成对抗扰动攻击"""
        attack_samples = []

        for sample in self.honest_samples:
            for eps in self.epsilon:
                x_perturbed = self.fgsm_perturbation(sample.x_curr.copy(), eps)

                attack_sample = Sample(
                    sample_id=sample.sample_id,
                    layer_idx=sample.layer_idx,
                    x_prev=sample.x_prev.copy(),
                    x_curr=x_perturbed,
                    label=f'adversarial_perturbation_eps_{eps}',
                    metadata={
                        **sample.metadata,
                        'attack_type': 'adversarial_perturbation',
                        'epsilon': eps
                    }
                )
                attack_samples.append(attack_sample)

        logger.info(f"生成了 {len(attack_samples)} 个对抗扰动攻击样本")
        return attack_samples

