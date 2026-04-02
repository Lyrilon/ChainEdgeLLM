"""
Data Generator Module
生成诚实样本和攻击样本用于 Stage 1 可分性测试
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """样本数据结构"""
    sample_id: int
    layer_idx: int
    x_prev: np.ndarray  # 输入隐藏状态 (前一层的输出)
    x_curr: np.ndarray  # 当前隐藏状态 (当前层的输出)
    label: str          # 'honest', 'random_noise', 'replay', 'layer_skip', etc.
    metadata: Dict      # 额外信息


class HonestSampleGenerator:
    """
    诚实样本生成器
    通过正常前向传播生成 (x_{k-1}, x_k) 对
    """

    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _get_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        获取输入的 embedding，支持多种模型架构

        Args:
            input_ids: 输入 token IDs

        Returns:
            embedding 张量
        """
        # 1. GPT-2 风格
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            return self.model.transformer.wte(input_ids)

        # 2. Qwen2 / LLaMA / Mistral / Gemma 风格
        if hasattr(self.model, 'model'):
            # 尝试多种可能的 embedding 属性
            if hasattr(self.model.model, 'embed_tokens'):
                return self.model.model.embed_tokens(input_ids)
            if hasattr(self.model.model, 'embeddings'):
                return self.model.model.embeddings(input_ids)
            if hasattr(self.model.model, 'word_embeddings'):
                return self.model.model.word_embeddings(input_ids)
            # 尝试多层嵌套
            if hasattr(self.model.model, 'model'):
                if hasattr(self.model.model.model, 'embed_tokens'):
                    return self.model.model.model.embed_tokens(input_ids)

        # 3. BERT / RoBERTa 风格
        if hasattr(self.model, 'embeddings'):
            return self.model.embeddings(input_ids)

        # 4. 通用查找
        for attr_name in ['embed_tokens', 'embeddings', 'word_embeddings', 'wte', 'embedding']:
            if hasattr(self.model, attr_name):
                embed_layer = getattr(self.model, attr_name)
                if callable(embed_layer):
                    return embed_layer(input_ids)
                return embed_layer(input_ids)

        # 5. 最后的尝试：使用模型的 get_input_embeddings 方法
        if hasattr(self.model, 'get_input_embeddings'):
            embed_layer = self.model.get_input_embeddings()
            return embed_layer(input_ids)

        raise ValueError(f"无法找到模型的 embedding 层，请检查模型架构")

    def generate(self, texts: List[str], target_layers: List[int]) -> List[Sample]:
        """
        生成诚实样本

        Args:
            texts: 输入文本列表
            target_layers: 目标层索引列表

        Returns:
            样本列表
        """
        from model_loader import HiddenStateExtractor

        samples = []
        extractor = HiddenStateExtractor(self.model, target_layers)

        for idx, text in enumerate(tqdm(texts, desc="生成诚实样本")):
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True
            ).to(self.device)

            # 提取隐藏状态
            hidden_states = extractor.extract(
                inputs['input_ids'],
                inputs.get('attention_mask')
            )

            # 为每个目标层创建样本
            for layer_idx in target_layers:
                if layer_idx not in hidden_states:
                    continue

                # 获取当前层输出
                x_curr = hidden_states[layer_idx].numpy()

                # 获取前一层输出 (对于第0层，使用 embedding)
                if layer_idx == 0:
                    with torch.no_grad():
                        x_prev = self._get_embedding(inputs['input_ids']).cpu().numpy()
                else:
                    # 需要重新提取前一层的输出
                    prev_extractor = HiddenStateExtractor(self.model, [layer_idx - 1])
                    prev_hidden = prev_extractor.extract(
                        inputs['input_ids'],
                        inputs.get('attention_mask')
                    )
                    x_prev = prev_hidden.get(layer_idx - 1, x_curr).numpy()

                # 对每个序列位置创建样本 (取平均或第一个token)
                # 这里我们取序列的平均池化
                x_prev_pooled = np.mean(x_prev[0], axis=0)  # [hidden_dim]
                x_curr_pooled = np.mean(x_curr[0], axis=0)  # [hidden_dim]

                sample = Sample(
                    sample_id=idx,
                    layer_idx=layer_idx,
                    x_prev=x_prev_pooled,
                    x_curr=x_curr_pooled,
                    label='honest',
                    metadata={
                        'text': text[:100],  # 存储前100字符
                        'seq_length': x_curr.shape[1],
                        'hidden_dim': x_curr.shape[2]
                    }
                )
                samples.append(sample)

        logger.info(f"生成了 {len(samples)} 个诚实样本")
        return samples


class AttackSampleGenerator:
    """
    攻击样本生成器
    生成各种类型的攻击样本
    """

    def __init__(self, honest_samples: List[Sample], seed: int = 42):
        self.honest_samples = honest_samples
        self.rng = np.random.RandomState(seed)

    def generate_random_noise(self, noise_levels: List[float]) -> List[Sample]:
        """
        生成随机噪声攻击样本 (Byzantine Failure)

        Args:
            noise_levels: 噪声标准差列表

        Returns:
            攻击样本列表
        """
        attack_samples = []

        for sample in tqdm(self.honest_samples, desc="生成随机噪声样本"):
            for noise_level in noise_levels:
                # 生成高斯噪声
                noise = self.rng.normal(
                    loc=0.0,
                    scale=noise_level,
                    size=sample.x_curr.shape
                ).astype(np.float32)

                # 纯噪声替换：Byzantine Failure - 完全替换合法状态
                x_malicious = noise

                attack_sample = Sample(
                    sample_id=sample.sample_id,
                    layer_idx=sample.layer_idx,
                    x_prev=sample.x_prev.copy(),
                    x_curr=x_malicious,
                    label=f'random_noise_sigma_{noise_level}',
                    metadata={
                        **sample.metadata,
                        'attack_type': 'random_noise',
                        'noise_level': noise_level,
                        'noise_norm': float(np.linalg.norm(noise))
                    }
                )
                attack_samples.append(attack_sample)

        logger.info(f"生成了 {len(attack_samples)} 个随机噪声样本")
        return attack_samples

    def generate_replay_attacks(self, replay_types: List[str], pool_size: int = 500) -> List[Sample]:
        """
        生成重放攻击样本 (Replay Attack)

        Args:
            replay_types: 重放类型列表
            pool_size: 重放池大小

        Returns:
            攻击样本列表
        """
        attack_samples = []

        # 构建重放池
        replay_pool = self.honest_samples.copy()
        if len(replay_pool) > pool_size:
            replay_pool = self.rng.choice(replay_pool, pool_size, replace=False).tolist()

        for sample in tqdm(self.honest_samples, desc="生成重放样本"):
            for replay_type in replay_types:
                # 根据重放类型选择重放目标
                candidates = replay_pool

                if replay_type == 'cross_sequence_same_layer':
                    # 不同序列，同层
                    candidates = [s for s in candidates
                                  if s.sample_id != sample.sample_id
                                  and s.layer_idx == sample.layer_idx]

                elif replay_type == 'cross_sequence_cross_layer':
                    # 不同序列，不同层
                    candidates = [s for s in candidates
                                  if s.sample_id != sample.sample_id
                                  and s.layer_idx != sample.layer_idx]

                elif replay_type == 'same_sequence_cross_layer':
                    # 同序列，不同层
                    candidates = [s for s in candidates
                                  if s.sample_id == sample.sample_id
                                  and s.layer_idx != sample.layer_idx]

                if not candidates:
                    continue

                # 随机选择一个重放目标
                replay_target = self.rng.choice(candidates)

                attack_sample = Sample(
                    sample_id=sample.sample_id,
                    layer_idx=sample.layer_idx,
                    x_prev=sample.x_prev.copy(),
                    x_curr=replay_target.x_curr.copy(),
                    label=f'replay_{replay_type}',
                    metadata={
                        **sample.metadata,
                        'attack_type': 'replay',
                        'replay_type': replay_type,
                        'replay_from_sample_id': replay_target.sample_id,
                        'replay_from_layer': replay_target.layer_idx
                    }
                )
                attack_samples.append(attack_sample)

        logger.info(f"生成了 {len(attack_samples)} 个重放攻击样本")
        return attack_samples

    def generate_layer_skipping(self) -> List[Sample]:
        """
        生成层跳过攻击样本 (Identity Forgery)
        注意: Stage 1 无法有效检测，但生成用于完整性

        Returns:
            攻击样本列表
        """
        attack_samples = []

        for sample in tqdm(self.honest_samples, desc="生成层跳过样本"):
            # 直接返回输入作为输出: x_k = x_{k-1}
            attack_sample = Sample(
                sample_id=sample.sample_id,
                layer_idx=sample.layer_idx,
                x_prev=sample.x_prev.copy(),
                x_curr=sample.x_prev.copy(),  # 复制输入作为输出
                label='layer_skip',
                metadata={
                    **sample.metadata,
                    'attack_type': 'layer_skip',
                    'skipped_layer': sample.layer_idx
                }
            )
            attack_samples.append(attack_sample)

        logger.info(f"生成了 {len(attack_samples)} 个层跳过样本")
        return attack_samples

    def generate_precision_downgrade(self, target_bits: List[int] = [4, 8]) -> List[Sample]:
        """
        生成精度降级攻击样本
        注意: Stage 1 无法有效检测，但生成用于完整性

        Args:
            target_bits: 目标量化位数

        Returns:
            攻击样本列表
        """
        attack_samples = []

        for sample in tqdm(self.honest_samples, desc="生成精度降级样本"):
            for bits in target_bits:
                # 模拟量化-反量化
                x_float = sample.x_curr.copy()

                # 计算量化范围
                x_min, x_max = x_float.min(), x_float.max()

                # 量化到目标位数
                levels = 2 ** bits - 1
                x_quantized = np.round((x_float - x_min) / (x_max - x_min) * levels)

                # 反量化
                x_dequantized = x_quantized / levels * (x_max - x_min) + x_min

                attack_sample = Sample(
                    sample_id=sample.sample_id,
                    layer_idx=sample.layer_idx,
                    x_prev=sample.x_prev.copy(),
                    x_curr=x_dequantized.astype(np.float32),
                    label=f'precision_downgrade_{bits}bit',
                    metadata={
                        **sample.metadata,
                        'attack_type': 'precision_downgrade',
                        'original_bits': 32,
                        'target_bits': bits,
                        'quantization_error': float(np.mean(np.abs(x_float - x_dequantized)))
                    }
                )
                attack_samples.append(attack_sample)

        logger.info(f"生成了 {len(attack_samples)} 个精度降级样本")
        return attack_samples


class DatasetLoader:
    """
    数据集加载器 (使用 ModelScope)
    """

    # ModelScope 数据集映射
    DATASET_MAP = {
        "wikitext": "AI-ModelScope/wikitext-2-raw-v1",
        "wikitext-103": "AI-ModelScope/wikitext-103-raw-v1",
        "c4": "AI-ModelScope/c4",
        "openwebtext": "AI-ModelScope/openwebtext",
        # 指令微调数据集（推荐）
        "alpaca": "AI-ModelScope/alpaca-gpt4-data",
        "alpaca-cleaned": "AI-ModelScope/alpaca-data-cleaned",
        "dolly": "AI-ModelScope/dolly-15k",
        "sharegpt": "AI-ModelScope/sharegpt-json",
    }

    def __init__(self, dataset_name: str, cache_dir: str = "./data/cache"):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir

    def _get_modelscope_id(self, name: str) -> str:
        """获取 ModelScope 数据集 ID"""
        return self.DATASET_MAP.get(name, name)

    def load(self, split: str = "train", num_samples: int = 1000):
        """
        加载数据集

        Args:
            split: 数据分割 (train/validation/test)
            num_samples: 采样数量

        Returns:
            文本列表
        """
        try:
            from modelscope.msdatasets import MsDataset

            dataset_id = self._get_modelscope_id(self.dataset_name)
            logger.info(f"正在从 ModelScope 加载数据集: {dataset_id}")

            dataset = MsDataset.load(
                dataset_id,
                split=split,
                cache_dir=self.cache_dir
            )

            # 提取文本
            texts = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break

                # 处理不同数据集格式
                text = None

                # 1. 标准文本格式 (WikiText, C4, etc.)
                if 'text' in item:
                    text = item['text']
                elif 'content' in item:
                    text = item['content']

                # 2. 指令微调格式 (Alpaca, Dolly, etc.)
                elif 'instruction' in item:
                    # Alpaca 格式: instruction + input + output
                    instruction = item.get('instruction', '')
                    input_text = item.get('input', '')
                    output_text = item.get('output', '')

                    # 组合成完整对话
                    if input_text:
                        text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"
                    else:
                        text = f"Instruction: {instruction}\nOutput: {output_text}"

                # 3. ShareGPT 格式
                elif 'conversations' in item:
                    conversations = item['conversations']
                    if isinstance(conversations, list):
                        text = "\n".join([
                            f"{conv.get('from', 'user')}: {conv.get('value', '')}"
                            for conv in conversations
                        ])

                # 4. Dolly 格式
                elif 'context' in item and 'response' in item:
                    context = item.get('context', '')
                    instruction = item.get('instruction', '')
                    response = item.get('response', '')
                    text = f"Context: {context}\nInstruction: {instruction}\nResponse: {response}"

                else:
                    text = str(item)

                # 过滤空文本和短文本
                if text and len(text.strip()) > 50:
                    texts.append(text.strip())

            logger.info(f"成功加载 {len(texts)} 条文本")
            return texts

        except ImportError as e:
            logger.error(f"缺少依赖: {e}")
            logger.error("请运行: pip install addict")
            raise RuntimeError(f"缺少必要依赖: {e}") from e
        except Exception as e:
            logger.error(f"从 ModelScope 加载数据集失败: {e}")
            raise RuntimeError(f"数据集加载失败: {e}") from e


def test_data_generator():
    """测试数据生成器"""
    # 创建一些模拟样本
    hidden_dim = 768
    num_samples = 10

    honest_samples = []
    for i in range(num_samples):
        x_prev = np.random.randn(hidden_dim).astype(np.float32)
        # 模拟残差连接: x_curr = x_prev + small_delta
        delta = np.random.randn(hidden_dim).astype(np.float32) * 0.1
        x_curr = x_prev + delta

        sample = Sample(
            sample_id=i,
            layer_idx=4,
            x_prev=x_prev,
            x_curr=x_curr,
            label='honest',
            metadata={}
        )
        honest_samples.append(sample)

    # 测试攻击生成器
    attack_gen = AttackSampleGenerator(honest_samples)

    noise_samples = attack_gen.generate_random_noise([0.5, 1.0])
    print(f"噪声样本数: {len(noise_samples)}")

    replay_samples = attack_gen.generate_replay_attacks(['cross_sequence_same_layer'])
    print(f"重放样本数: {len(replay_samples)}")

    skip_samples = attack_gen.generate_layer_skipping()
    print(f"层跳过样本数: {len(skip_samples)}")


if __name__ == "__main__":
    test_data_generator()
